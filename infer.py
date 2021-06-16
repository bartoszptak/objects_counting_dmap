"""This script apply a chosen model on a given image.

One needs to choose a network architecture and provide the corresponding
state dictionary.

Example:

    $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg

The script also allows to visualize the results by drawing a resulting
density map on the input image.

Example:

    $ $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg --visualize

"""
import os

import click
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from PIL import Image
import segmentation_models_pytorch as smp
import cv2
from glob import glob

from model import UNet, FCRN_A

def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

@click.command()
@click.option('-i', '--image_path',
              type=click.File('r'),
              required=False,
              help="A path to an input image.")
@click.option('-n', '--network_architecture',
              required=True,
              help='Model architecture.')
@click.option('-c', '--checkpoint',
              type=click.File('r'),
              required=True,
              help='A path to a checkpoint with weights.')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--one_channel',
              is_flag=True,
              help="Turn this on for one channel images (required for ucsd).")
@click.option('--pad',
              is_flag=True,
              help="Turn on padding for input image (required for ucsd).")
@click.option('--visualize',
              is_flag=True,
              help="Visualize predicted density map.")
@click.option('--flow', type=click.Choice(['', 'median', 'dis', 'dis2']), default='', help='')
@click.option('-s', '--seq',
              type=str,
              required=False,
              help="A path to an input seq.")
@click.option('--to_onnx',
              is_flag=True,
              help="")          
              
def infer(image_path: str,
          network_architecture: str,
          checkpoint: str,
          unet_filters: int,
          convolutions: int,
          one_channel: bool,
          pad: bool,
          visualize: bool,
          flow: str,
          to_onnx: bool,
          seq: str):
    """Run inference for a single image."""
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # only UCSD dataset provides greyscale images instead of RGB
    in_channels = 3
    if flow == 'median':
        in_channels += 1
    elif flow == 'dis':
        in_channels += 2
    elif flow == 'dis2':
        in_channels += 5

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

    # network = torch.nn.DataParallel()
    network = network.to(device)

    # load provided state dictionary
    # note: by default train.py saves the model in data parallel mode
    network.load_state_dict(torch.load(checkpoint.name, map_location=torch.device('cpu')))
    network.eval()

    if to_onnx:
        
        input_var = torch.rand(1, 5, 608, 608)
        torch.onnx.export(network.module.to(torch.device('cpu')), input_var, 'visdrone_unet++_resnet34_11.onnx', input_names=["input"], output_names=["output"], verbose=False, export_params=True, opset_version=11)
        exit(0)

    if flow == 'dis' or flow == 'dis2':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        inst.setUseSpatialPropagation(False)

        flow_img = None
        prevgray = None

        for i, path in enumerate(sorted(glob(seq+'*'))):

            image = Image.open(path)
            img = np.array(image, dtype=np.float32)
            y, x = img.shape[:2]

            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            if prevgray is None:
                prevgray = gray.copy()
            
            if flow_img is not None:
                flow_img = inst.calc(prevgray, gray, warp_flow(flow_img,flow_img))
            else:
                flow_img = inst.calc(prevgray, gray, None)

            if flow == 'dis2':
                sub = gray-prevgray

                fx, fy = flow_img[:,:,0], flow_img[:,:,1]
                ang = np.arctan2(fy, fx) + np.pi
                ang = ang*(180/np.pi/2)*255*180

                v = np.sqrt(fx*fx+fy*fy)
                v = np.minimum(v*4, 255)

                img = np.concatenate((
                    img,
                    np.reshape(sub*255., (*img.shape[:2], 1)),
                    np.reshape(flow_img*255., (*img.shape[:2], 2)),
                    np.reshape(ang, (*img.shape[:2], 1)),
                    np.reshape(v, (*img.shape[:2], 1)),
                    ), axis=2)
            else:
                img = np.concatenate((img, np.reshape(flow_img*255., (*img.shape[:2], 2))), axis=2)

            prevgray = gray

            img = cv2.resize(img, (608,608))
            img = img * 1./255.

            # with open(f'test_{i}.npy', 'wb') as f:
            #     np.save(f, img)

            density_map = network(TF.to_tensor(img).unsqueeze_(0))

            # note: density maps were normalized to 100 * no. of objects
            n_objects = torch.sum(density_map).item()/100

            print(f"The number of objects found: {n_objects}")

            if visualize:
                _visualize(image.resize((608,608)), density_map.squeeze().cpu().detach().numpy())

    else:

        img = Image.open(image_path.name).resize((608,608))

        # padding was applied for ucsd images to allow down and upsampling
        if pad:
            img = Image.fromarray(np.pad(img, 1, 'constant', constant_values=0))

        # network's output represents a density map
        density_map = network(TF.to_tensor(img).unsqueeze_(0))

        # note: density maps were normalized to 100 * no. of objects
        n_objects = torch.sum(density_map).item()/100

        print(f"The number of objects found: {n_objects}")

        if visualize:
            _visualize(img, density_map.squeeze().cpu().detach().numpy())


def _visualize(img, dmap):

    plt.imshow(dmap)
    plt.show()

    print(img.size, dmap.shape)

    """Draw a density map onto the image."""
    # keep the same aspect ratio as an input image
    fig, ax = plt.subplots(figsize=figaspect(1.0 * img.size[1] / img.size[0]))
    fig.subplots_adjust(0, 0, 1, 1)

    # plot a density map without axis
    ax.imshow(dmap, cmap="hot")
    plt.axis('off')
    fig.canvas.draw()

    # create a PIL image from a matplotlib figure
    dmap = Image.frombytes('RGB',
                           fig.canvas.get_width_height(),
                           fig.canvas.tostring_rgb())

    # add a alpha channel proportional to a density map value
    dmap.putalpha(dmap.convert('L'))

    # display an image with density map put on top of it
    Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size)).show()

    


if __name__ == "__main__":
    infer()
