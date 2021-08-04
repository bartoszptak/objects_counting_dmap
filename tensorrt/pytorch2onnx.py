import click
import torch
import torch.onnx
import segmentation_models_pytorch as smp


@click.command()
@click.option('--checkpoint_path', help='Path to PyTorch checkpoint', default='./visdrone_dis_unet_resnet18_adam_aug_mosaic_flow_dis.pth')
@click.option('--model_architecture', help='Model architecture, available: {UNet, UNet++}', default='UNet')
@click.option('--encoder', help='Model encoder name', default='resnet34')
@click.option('--in_channels', help='Number of input channels', default=5)
@click.option('--input_size', help='Model input size', default=608)
def convert(checkpoint_path, model_architecture, encoder, in_channels, input_size):
    print('Converting PyTorch checkpoint to ONNX model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if model_architecture == 'UNet':
        network = smp.Unet(encoder_name=encoder, in_channels=in_channels, classes=1)
    elif model_architecture == 'UNet++':
        network = smp.UnetPlusPlus(encoder_name=encoder, in_channels=in_channels, classes=1)
        
    network = network.to(device)

    network.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    input_var = torch.rand(1, in_channels, input_size, input_size)
    onnx_path = checkpoint_path.replace('pth', 'onnx')

    torch.onnx.export(network.to(torch.device('cpu')), input_var, onnx_path, input_names=["input"], output_names=["output"], verbose=False, export_params=True, opset_version=11)
    print(f'Model converted and saved as {onnx_path}')


if __name__ == '__main__':
    convert()