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
@click.option('--name', default='',
              required=True,
              help='')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--plot', is_flag=True, help="Generate a live plot.")
@click.option('--loss', type=click.Choice(['mse', 'weight']), default='mse')
@click.option('-c', '--checkpoint',
              type=click.File('r'),
              required=False,
              default=None,
              help='A path to a checkpoint with weights.')
@click.option('--sliced', default=False, is_flag=True, help='')
def eval(dataset_name: str,
          network_architecture: str,
	      name: str,
          loss: str,
          sliced: bool,
          unet_filters: int,
          convolutions: int,
          checkpoint: str,
          plot: bool):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.join(dataset_name, f"valid.h5")
    dataset= H5Dataset(data_path, mosaic = False, aug = False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == 'ucsd' else 3

    # initialize a model based on chosen network_architecture

    if network_architecture in ['UNet', 'FCRN_A']:
        network = {
            'UNet': UNet,
            'FCRN_A': FCRN_A
        }[network_architecture](input_filters=input_channels,
                                filters=unet_filters,
                                N=convolutions)

    elif network_architecture[:5] == 'UNet_':
        network = smp.Unet(encoder_name=network_architecture.split('_')[-1], in_channels=3, classes=1) 
    elif network_architecture[:7] == 'UNet++_':
        network = smp.UnetPlusPlus(encoder_name=network_architecture.split('_')[-1], in_channels=3, classes=1) 
    elif network_architecture[:4] == 'FPN_':
        network = smp.FPN(encoder_name=network_architecture.split('_')[-1], in_channels=3, classes=1) 
    elif network_architecture[:4] == 'PSP_':
        network = smp.PSPNet(encoder_name=network_architecture.split('_')[-1], in_channels=3, classes=1)
    else:
        raise NotImplementedError

    network = torch.nn.DataParallel(network.to(device))

    if checkpoint:
        #if device == 'cpu':
        #    print('cpu')
        #    network.load_state_dict(torch.load(checkpoint.name, map_location=torch.device('cpu')))
        #else:
        #    network.load_state_dict(torch.load(checkpoint.name))

        network.load_state_dict(torch.load(checkpoint.name, map_location=torch.device('cpu')))
        network.eval()

    # initialize loss, optimized and learning rate scheduler

    if loss == 'weight':
        def count_loss(output, target):
            mae = torch.nn.MSELoss()
            loss = torch.mean(torch.absolute(output.sum() - target.sum()))/10e4
            return 0.9 * mae(output, target) + 0.1 * loss

        loss_fn = count_loss
    else:
        loss_fn = torch.nn.MSELoss()

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    valid_looper = Looper(network, device, loss_fn, None,
                          dataloader, len(dataset), plots[1],
                          validation=True, sliced=sliced)

    with torch.no_grad():
        result = valid_looper.run()

    print(f"[Training done] Best result: {result}, dataset: {dataset_name}, model: {network_architecture}, aug: {aug}, mosaic: {mosaic}, sliced: {sliced}")

if __name__ == '__main__':
    eval()
