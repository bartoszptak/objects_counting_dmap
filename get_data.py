"""A tool to download and preprocess data, and generate HDF5 file.

Available datasets:

    * cell: http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html
    * mall: http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
    * ucsd: http://www.svcl.ucsd.edu/projects/peoplecnt/
"""
import os
import shutil
import zipfile
from glob import glob
from typing import List, Tuple
import pandas as pd
pd.set_option('mode.chained_assignment',None)
from glob import glob
from tqdm import tqdm
import json
import numpy as np
import cv2

import click
import h5py
import wget
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


@click.command()
@click.option('--dataset',
              type=click.Choice(['cell', 'mall', 'ucsd', 'visdrone', 'gcc']),
              required=True)
@click.option('--sliced', default=False, is_flag=True, help='')
@click.option('--flow', type=click.Choice(['', 'median', 'dis', 'dis2']), default='', help='')
def get_data(dataset: str, sliced: bool, flow: str):
    """
    Get chosen dataset and generate HDF5 files with training
    and validation samples.
    """
    # dictionary-based switch statement
    if sliced or flow!='':
        {
        'visdrone': generate_visdrone_data,
    }[dataset](sliced=sliced, flow=flow)
    else:
        {
        'cell': generate_cell_data,
        'mall': generate_mall_data,
        'ucsd': generate_ucsd_data,
        'visdrone': generate_visdrone_data,
        'gcc': generate_gcc_data
    }[dataset]()


def create_hdf5(dataset_name: str,
                train_size: int,
                valid_size: int,
                img_size: Tuple[int, int],
                in_channels: int=3,
                whc=False,
                flow=''):
    """
    Create empty training and validation HDF5 files with placeholders
    for images and labels (density maps).

    Note:
    Datasets are saved in [dataset_name]/train.h5 and [dataset_name]/valid.h5.
    Existing files will be overwritten.

    Args:
        dataset_name: used to create a folder for train.h5 and valid.h5
        train_size: no. of training samples
        valid_size: no. of validation samples
        img_size: (width, height) of a single image / density map
        in_channels: no. of channels of an input image

    Returns:
        A tuple of pointers to training and validation HDF5 files.
    """

    if flow != '':
        dataset_name += f'_{flow}' 
    # create output folder if it does not exist
    os.makedirs(dataset_name, exist_ok=True)

    # create HDF5 files: [dataset_name]/(train | valid).h5
    train_h5 = h5py.File(os.path.join(dataset_name, 'train.h5'), 'w')
    valid_h5 = h5py.File(os.path.join(dataset_name, 'valid.h5'), 'w')

    if not whc:
        # add two HDF5 datasets (images and labels) for each HDF5 file
        for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
            h5.create_dataset('images', (size, in_channels, *img_size))
            h5.create_dataset('labels', (size, 1, *img_size))
    else:
        for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
            h5.create_dataset('images', (size, *img_size, in_channels))
            h5.create_dataset('labels', (size, *img_size, 1))

    return train_h5, valid_h5


def generate_label(label_info: np.array, image_shape: List[int]):
    """
    Generate a density map based on objects positions.

    Args:
        label_info: (x, y) objects positions
        image_shape: (width, height) of a density map to be generated

    Returns:
        A density map.
    """
    # create an empty density map
    label = np.zeros(image_shape, dtype=np.float32)

    # loop over objects positions and marked them with 400 on a label
    # note: *_ because some datasets contain more info except x, y coordinates
    for x, y, *_ in label_info:
        if y < image_shape[0] and x < image_shape[1]:
            label[int(y)][int(x)] = 100

    # apply a convolution with a Gaussian kernel
    label = gaussian_filter(label, sigma=(1, 1), order=0)

    return label


def get_and_unzip(url: str, location: str="."):
    """Extract a ZIP archive from given URL.

    Args:
        url: url of a ZIP file
        location: target location to extract archive in
    """
    dataset = wget.download(url)
    dataset = zipfile.ZipFile(dataset)
    dataset.extractall(location)
    dataset.close()
    os.remove(dataset.filename)
    

def generate_ucsd_data():
    """Generate HDF5 files for mall dataset."""
    # download and extract data
    get_and_unzip(
        'http://www.svcl.ucsd.edu/projects/peoplecnt/db/ucsdpeds.zip'
    )
    # download and extract annotations
    get_and_unzip(
        'http://www.svcl.ucsd.edu/projects/peoplecnt/db/vidf-cvpr.zip'
    )
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5('ucsd',
                                     train_size=1500,
                                     valid_size=500,
                                     img_size=(160, 240),
                                     in_channels=1)

    def fill_h5(h5, labels, video_id, init_frame=0, h5_id=0):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            labels: the list of labels
            video_id: the id of a scene
            init_frame: the first frame in given list of labels
            h5_id: next dataset id to be used
        """
        video_name = f"vidf1_33_00{video_id}"
        video_path = f"ucsdpeds/vidf/{video_name}.y/"

        for i, label in enumerate(labels, init_frame):
            # path to the next frame (convention: [video name]_fXXX.jpg)
            img_path = f"{video_path}/{video_name}_f{str(i+1).zfill(3)}.png"

            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            # generate a density map by applying a Gaussian filter
            label = generate_label(label[0][0][0], image.shape)

            # pad images to allow down and upsampling
            image = np.pad(image, 1, 'constant', constant_values=0)
            label = np.pad(label, 1, 'constant', constant_values=0)

            # save data to HDF5 file
            h5['images'][h5_id + i - init_frame, 0] = image
            h5['labels'][h5_id + i - init_frame, 0] = label

    # dataset contains 10 scenes
    for scene in range(10):
        # load labels infomation from provided MATLAB file
        # it is numpy array with (x, y) objects position for subsequent frames
        descriptions = loadmat(f'vidf-cvpr/vidf1_33_00{scene}_frame_full.mat')
        labels = descriptions['frame'][0]

        # use first 150 frames for training and the last 50 for validation
        # start filling from the place last scene finished
        fill_h5(train_h5, labels[:150], scene, 0, 150 * scene)
        fill_h5(valid_h5, labels[150:], scene, 150, 50 * scene)

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

    # cleanup
    shutil.rmtree('ucsdpeds')
    shutil.rmtree('vidf-cvpr')


def generate_mall_data():
    """Generate HDF5 files for mall dataset."""
    # download and extract dataset
    get_and_unzip(
        'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip'
    )
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5('mall',
                                     train_size=1500,
                                     valid_size=500,
                                     img_size=(480, 640),
                                     in_channels=3)

    # load labels infomation from provided MATLAB file
    # it is a numpy array with (x, y) objects position for subsequent frames
    labels = loadmat('mall_dataset/mall_gt.mat')['frame'][0]

    def fill_h5(h5, labels, init_frame=0):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            labels: the list of labels
            init_frame: the first frame in given list of labels
        """
        for i, label in enumerate(labels, init_frame):
            # path to the next frame (filename convention: seq_XXXXXX.jpg)
            img_path = f"mall_dataset/frames/seq_{str(i+1).zfill(6)}.jpg"

            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            image = np.transpose(image, (2, 0, 1))

            # generate a density map by applying a Gaussian filter
            label = generate_label(label[0][0][0], image.shape[1:])

            # save data to HDF5 file
            h5['images'][i - init_frame] = image
            h5['labels'][i - init_frame, 0] = label

    # use first 1500 frames for training and the last 500 for validation
    fill_h5(train_h5, labels[:1500])
    fill_h5(valid_h5, labels[1500:], 1500)

    # close HDF5 file
    train_h5.close()
    valid_h5.close()

    # cleanup
    shutil.rmtree('mall_dataset')

def generate_visdrone_data(sliced=False, flow=''):
    """Generate HDF5 files for visdrone dataset."""
    # download and extract dataset

    bpath = '../VisDrone2020-CC/'
    img_size = (608, 608)

    with open(bpath + 'trainlist.txt') as f:
        train_list = sorted(f.read().split('\n'))

    test_list = train_list[66:]
    train_list = train_list[:66]

    train_size = sum([len(glob(bpath + f'sequences/{l}/*')) for l in train_list])
    test_size = sum([len(glob(bpath + f'sequences/{l}/*')) for l in test_list])

    if sliced:
        size = 608
        padding = size-32
        h, w = 1080, 1920

        xs = sorted(set([x if (w-x)>size else (w-size) for x in range(0, w, padding)]))
        xs = xs[:-1] if len(xs) > 1 and xs[-1]-xs[-2] > size else xs
        xs = np.array(xs)
        xs[xs<0] = 0

        ys = sorted(set([y if (h-y)>size else (h-size) for y in range(0, h, padding)]))
        ys = ys[:-1] if len(ys) > 1 and ys[-1]-ys[-2] > size else ys
        ys = np.array(ys)
        ys[ys<0] = 0

        print(xs, ys)

        train_size *= len(xs) * len(ys)
        test_size *= len(xs) * len(ys)

    if flow == 'median':
        in_channels = 4
    elif flow == 'dis':
        in_channels = 5
    elif flow == 'dis2':
        in_channels = 6
    else:
        in_channels = 3
    
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5('visdrone',
                                     train_size=train_size,
                                     valid_size=test_size,
                                     img_size=img_size,
                                     in_channels=in_channels,
                                     whc=True, flow = flow)

    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

    def fill_h5(h5, labels, init_frame=0, slide = None, flow=''):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            labels: the list of labels
            init_frame: the first frame in given list of labels
        """
        i = init_frame
        for seq in tqdm(labels):
            if flow == 'median':
                backSub = cv2.createBackgroundSubtractorMOG2()

                previous_frame = None
                current_frame = None
                next_frame = None
            elif flow == 'dis' or flow == 'dis2':
                inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                inst.setUseSpatialPropagation(False)

                flow_img = None
                prevgray = None

            df_lab = pd.read_csv(bpath + f'annotations/{seq}.txt', names=['img', 'x', 'y'])

            for path in sorted(glob(bpath + f'sequences/{seq}/*')):
                # get an image as numpy array
                image = Image.open(path)
                x, y = image.size
                
                image = np.array(image, dtype=np.float32)

                if flow == 'median':
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                    if previous_frame is None:
                        previous_frame = gray.copy()
                    if current_frame is None:
                        current_frame = gray.copy()

                    next_frame = gray.copy()

                    I1 = cv2.absdiff(current_frame, next_frame)
                    I2 = cv2.absdiff(previous_frame, next_frame)

                    fg = cv2.bitwise_and(I1, I2)
                    fg[fg<10] = 0

                    previous_frame = current_frame
                    current_frame = next_frame

                    image = np.concatenate((image, np.reshape(fg, (*image.shape[:2], 1))), axis=2)
                elif flow == 'dis' or flow == 'dis2':
                    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

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

                        image = np.concatenate((
                            image,
                            np.reshape(sub*255., (*image.shape[:2], 1)),
                            #np.reshape(flow_img*255., (*image.shape[:2], 2)),
                            np.reshape(ang, (*image.shape[:2], 1)),
                            np.reshape(v, (*image.shape[:2], 1)),
                            ), axis=2)
                    else:
                        image = np.concatenate((image, np.reshape(flow_img*255., (*image.shape[:2], 2))), axis=2)

                    prevgray = gray
                print(sliced, flow, image.shape)
                if not sliced:
                    image = cv2.resize(image, img_size[::-1])

                image = image * 1./255.

                index = int(path.split('/')[-1].split('.')[0])
                loc = df_lab[df_lab.img==index]

                if slide is not None:
                    for j, yy in enumerate(slide[0]):
                        for k, xx in enumerate(slide[1]):

                            part = image[yy:yy+slide[3], xx:xx+slide[3]]

                            loc_loc = loc[loc.img==index].copy()

                            loc_loc.loc[:, 'x'] -= xx
                            loc_loc.loc[:, 'y'] -= yy

                            loc_loc = loc_loc[(loc_loc.x < slide[3]) & (loc_loc.y < slide[3])]
                            loc_loc = loc_loc[(loc_loc.x > 0) & (loc_loc.y > 0)]

                            # generate a density map by applying a Gaussian filter
                            label = generate_label(loc_loc[['x','y']].values, part.shape[:2])

                            # save data to HDF5 file
                            h5['images'][i - init_frame] = part
                            h5['labels'][i - init_frame, :, :, 0] = label

                            i+=1

                else:
                    loc.loc[:, 'x'] *= img_size[1]/x
                    loc.loc[:, 'y'] *=img_size[0]/y

                    # generate a density map by applying a Gaussian filter
                    label = generate_label(loc[loc.img==index][['x','y']].values, image.shape[:2])

                    # save data to HDF5 file
                    h5['images'][i - init_frame] = image
                    h5['labels'][i - init_frame, :, :, 0] = label
                    
                    i+=1
         
    # use first 1500 frames for training and the last 500 for validation
    if sliced:
        fill_h5(train_h5, train_list, flow=flow, slide=(ys, xs, padding, size, flow))
        fill_h5(valid_h5, test_list, flow=flow, slide=(ys, xs, padding, size, flow))
    else:
        fill_h5(train_h5, train_list, flow=flow)
        fill_h5(valid_h5, test_list, flow=flow)

    # close HDF5 file
    train_h5.close()
    valid_h5.close()

def generate_gcc_data():
    """Generate HDF5 files for mall dataset."""
    # download and extract dataset

    bpath = '../GCC/'
    img_size = (608, 608)

    all_scenees = glob(bpath+'*')
    np.random.seed(0)
    np.random.shuffle(all_scenees)

    test_list = all_scenees[95:]
    train_list = all_scenees[:95]

    train_size = sum([len(glob(bpath + f'{l}/pngs/*')) for l in train_list])
    test_size = sum([len(glob(bpath + f'{l}/pngs/*')) for l in test_list])
    
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5('gcc',
                                     train_size=train_size,
                                     valid_size=test_size,
                                     img_size=img_size,
                                     in_channels=3,
                                     whc=True)

    def fill_h5(h5, labels, init_frame=0):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            labels: the list of labels
            init_frame: the first frame in given list of labels
        """
        i = init_frame
        for seq in tqdm(labels):
            for path in sorted(glob(bpath + f'{seq}/pngs/*')):

                # get an image as numpy array
                image = Image.open(path)
                x, y = image.size

                json_path = path.replace('pngs/', 'jsons/').replace('.png', '.json')

                with open(json_path, 'r') as f:
                    ann = json.load(f)['image_info']

                loc = pd.DataFrame(ann, columns=['y', 'x'])
                
                image = np.array(image.resize(img_size[::-1]), dtype=np.float32) / 255

                loc.loc[:, 'x'] *= img_size[1]/x
                loc.loc[:, 'y'] *=img_size[0]/y

                # generate a density map by applying a Gaussian filter
                label = generate_label(loc[['x','y']].values, image.shape[:2])

                # save data to HDF5 file
                h5['images'][i - init_frame] = image
                h5['labels'][i - init_frame, :, :, 0] = label
                
                i+=1
         
    # use first 1500 frames for training and the last 500 for validation
    fill_h5(train_h5, train_list)
    fill_h5(valid_h5, test_list)

    # close HDF5 file
    train_h5.close()
    valid_h5.close()


def generate_cell_data():
    """Generate HDF5 files for fluorescent cell dataset."""
    # download and extract dataset
    get_and_unzip(
        'http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip',
        location='cells'
    )
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5('cell',
                                     train_size=150,
                                     valid_size=50,
                                     img_size=(256, 256),
                                     in_channels=3)

    # get the list of all samples
    # dataset name convention: XXXcell.png (image) XXXdots.png (label)
    image_list = glob(os.path.join('cells', '*cell.*'))
    image_list.sort()

    def fill_h5(h5, images):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            images: the list of images paths
        """
        for i, img_path in enumerate(images):
            # get label path
            label_path = img_path.replace('cell.png', 'dots.png')
            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255
            image = np.transpose(image, (2, 0, 1))

            # convert a label image into a density map: dataset provides labels
            # in the form on an image with red dots placed in objects position

            # load an RGB image
            label = np.array(Image.open(label_path))
            # make a one-channel label array with 100 in red dots positions
            label = 100.0 * (label[:, :, 0] > 0)
            # generate a density map by applying a Gaussian filter
            label = gaussian_filter(label, sigma=(1, 1), order=0)

            # save data to HDF5 file
            h5['images'][i] = image
            h5['labels'][i, 0] = label

    # use first 150 samples for training and the last 50 for validation
    fill_h5(train_h5, image_list[:150])
    fill_h5(valid_h5, image_list[150:])

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

    # cleanup
    shutil.rmtree('cells')

if __name__ == '__main__':
    get_data()
