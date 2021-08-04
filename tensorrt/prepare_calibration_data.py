import os
import cv2
import click
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copyfile, rmtree


def _warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Remap the input optical flow image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    flow : np.ndarray
        Optical flow image.

    Returns
    -------
    np.ndarray
        Remapped optical flow image.
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


@click.command()
@click.option('-s', '--sequences_path', help='Path to VisDrone CC sequences', default='./sequences')
@click.option('-c', '--calibration_dataset_destination_path', help='Destination path of calibration dataset', default='./int8_calibration_dataset')
@click.option('-f', '--flow_type', help='DIS optical flow type, available: {medium, fast, ultrafast}', default='medium')
def main(sequences_path, calibration_dataset_destination_path, flow_type):
    if os.path.exists(calibration_dataset_destination_path):
        print(f'Directory {calibration_dataset_destination_path} exists. Removing...')
        rmtree(calibration_dataset_destination_path)

    print(f'Creating directory {calibration_dataset_destination_path} ...')
    os.makedirs(f'{calibration_dataset_destination_path}/imgs_flows/')
    os.makedirs(f'{calibration_dataset_destination_path}/imgs/')

    if flow_type == 'medium':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    elif flow_type == 'fast':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    elif flow_type == 'ultrafast':
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    else:
        print('Flow type unknown')
        inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    print('Set {} DIS flow type'.format(flow_type))
    inst.setUseSpatialPropagation(False)

    for dir in tqdm(os.listdir(sequences_path)):
        flow_img = None
        prevgray = None

        for filename in os.listdir(f'{sequences_path}/{dir}'):
            src_img_path = f'{sequences_path}/{dir}/{filename}'
            img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prevgray is None:
                prevgray = gray.copy()

            if flow_img is not None:
                flow_img = inst.calc(prevgray, gray, _warp_flow(flow_img, flow_img))
            else:
                flow_img = inst.calc(prevgray, gray, None)
            prevgray = gray
            
            np.save(f"{calibration_dataset_destination_path}/imgs_flows/{dir}_{filename.replace('.jpg', '.npy')}", flow_img.astype(np.float32))

            dst_img_path = f'{calibration_dataset_destination_path}/imgs/{dir}_{filename}'
            copyfile(src_img_path, dst_img_path)


if __name__ == '__main__':
    main()
