"""Main script used to train networks."""
import os
from typing import Union, Optional, List

import click
import torch
import numpy as np
from matplotlib import pyplot
import segmentation_models_pytorch as smp

from data_loader import H5Dataset
from looper import Looper
from model import UNet, FCRN_A


@click.command()
@click.option('-d', '--dataset_name',
              type=click.Choice(['cell', 'mall', 'ucsd', 'visdrone', 'gcc']),
              required=True,
              help='Dataset to train model on (expect proper HDF5 files).')
@click.option('-n', '--network_architecture',
              required=True,
              help='Model to train.')
@click.option('-lr', '--learning_rate', default=1e-2,
              help='Initial learning rate (lr_scheduler is applied).')
@click.option('-e', '--epochs', default=150, help='Number of training epochs.')
@click.option('--batch_size', default=8,
              help='Batch size for both training and validation dataloaders.')
@click.option('--aug', default=False, is_flag=True,
              help='')
@click.option('-e', '--eval', default=False, is_flag=True,
              help='')
@click.option('--sliced', default=False, is_flag=True,
              help='')
@click.option('--mosaic', default=False, is_flag=True,
              help='')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--plot', is_flag=True, help="Generate a live plot.")
@click.option('--loss', type=click.Choice(['mse', 'weight']), default='mse')
@click.option('--flow', type=click.Choice(['', 'median', 'dis']), default='', help='')
@click.option('-c', '--checkpoint',
              type=click.File('r'),
              required=False,
              default=None,
              help='A path to a checkpoint with weights.')
def train(dataset_name: str,
          network_architecture: str,
          learning_rate: float,
          epochs: int,
          batch_size: int,
          aug: bool,
          eval: bool,
          sliced: bool,
          loss: str,
          mosaic: bool,
          unet_filters: int,
          convolutions: int,
          checkpoint: str,
          plot: bool, 
          flow: str):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        # expected HDF5 files in dataset_name/(train | valid).h5
        data_path = os.path.join(dataset_name, f"{mode}.h5")
        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(data_path,
                                  flow = flow,
                                  mosaic = mosaic if mode == 'train' else False,
                                  aug = aug if mode == 'train' else False)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=batch_size)

    # only UCSD dataset provides greyscale images instead of RGB
    in_channels = 1 if dataset_name == 'ucsd' else 3
    if flow == 'median':
        in_channels += 1
    elif flow == 'dis':
        in_channels += 2

    # initialize a model based on chosen network_architecture

    if network_architecture in ['UNet', 'FCRN_A']:
        network = {
            'UNet': UNet,
            'FCRN_A': FCRN_A
        }[network_architecture](input_filters=in_channels,
                                filters=unet_filters,
                                N=convolutions)

    elif network_architecture[:5] == 'UNet_':
        network = smp.Unet(encoder_name=network_architecture.split('_')[-1], in_channels=in_channels, classes=1) 
    elif network_architecture[:7] == 'UNet++_':
        network = smp.UnetPlusPlus(encoder_name=network_architecture.split('_')[-1], in_channels=in_channels, classes=1) 
    elif network_architecture[:4] == 'FPN_':
        network = smp.FPN(encoder_name=network_architecture.split('_')[-1], in_channels=in_channels, classes=1) 
    elif network_architecture[:4] == 'PSP_':
        network = smp.PSPNet(encoder_name=network_architecture.split('_')[-1], in_channels=in_channels, classes=1)
    else:
        raise NotImplementedError

    network = torch.nn.DataParallel(network.to(device))

    if checkpoint:
        network.load_state_dict(torch.load(checkpoint.name))

    # initialize loss, optimized and learning rate scheduler

    if loss == 'weight':
        def count_loss(output, target):
            mae = torch.nn.MSELoss()
            loss = torch.mean(torch.absolute(output.sum() - target.sum()))/10e4
            return 0.9 * mae(output, target) + 0.1 * loss

        loss_fn = count_loss
    else:
        loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss_fn, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0])
    valid_looper = Looper(network, device, loss_fn, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True, sliced=sliced)

    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    if eval:
        epochs = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        if not eval:
            train_looper.run()
            lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update checkpoint if new best is reached
        if result < current_best and not eval:
            current_best = result
            path = f'{dataset_name}_{network_architecture}'
            if aug:
                path += '_aug'
            if mosaic:
                path += '_mosaic'
            if flow != '':
                path += '_flow'
            torch.save(network.state_dict(),
                       path+'.pth')

            print(f"\nNew best result: {result}")

        print("\n", "-"*80, "\n", sep='')

    if not eval:
        print(f"[Training done] Best result: {current_best}, dataset: {dataset_name}, model: {network_architecture}, aug: {aug}, mosaic: {mosaic}, flow: {flow}")

if __name__ == '__main__':
    train()
