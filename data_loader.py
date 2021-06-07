"""PyTorch dataset for HDF5 files generated with `get_data.py`."""
import os
import random
from typing import Optional

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from scipy.ndimage import gaussian_filter

class H5Dataset(Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(self,
                 dataset_path: str,
                 flow: str = '',
                 aug: bool = False,
                 mosaic: bool = False):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            
        """
        super(H5Dataset, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = self.h5['images']
        self.labels = self.h5['labels']
        self.aug = aug
        self.mosaic = mosaic
        self.flow = flow

        if self.aug:
            self.seq = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomGamma(gamma_limit=(150, 150), p=0.5),
            ])

        self.resize = A.Resize(608, 608, always_apply=True)

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def load_image(self, index: int):
        img, label = self.images[index], self.labels[index]

        if self.aug:
            transformed = self.seq(image=img[:,:,:3], mask=label)
            transformed["image"] = np.concatenate((transformed["image"], img[:,:,3:]), axis=2)
            transformed = self.resize(**transformed)
        else:
            transformed = self.resize(image=img, mask=label)

        # print(f'{label.sum()} : {transformed["mask"].sum()}')

        img, label = transformed["image"], transformed["mask"]

        return img, label

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
                

        if self.mosaic:
            img, label = self.load_mosaic(index=index)
        else:
            img, label = self.load_image(index=index)
        
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(18,10))
        # ax[0].imshow(img)
        # ax[1].imshow(label)
        # plt.show()

        return img.transpose((2,0,1)), label.transpose((2,0,1))
        

    def load_mosaic(self, index: int):
        s = 608
        xc, yc = [int(random.uniform(s * 0.25, s * 0.75)) for _ in range(2)]
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]

        if self.flow == 'median':
            in_channels = 4
        elif self.flow == 'dis':
            in_channels = 5
        else:
            in_channels = 3

        img4 = np.empty((s, s, in_channels), dtype=np.float32)
        labels4 = np.empty((s, s, 1), dtype=np.float32)

        masks = []

        for i, idx in enumerate(indices):
            img, lab = self.load_image(idx)

            if i == 0:  # top left
                xo, yo = int(random.uniform(0, s-xc)), int(random.uniform(0, s-yc))
                img4[:yc,:xc,:]    = img[yo:yo+yc,xo:xo+xc,:]
                labels4[:yc,:xc,0] = lab[yo:yo+yc,xo:xo+xc,0]

                masks.append(lab[yo:yo+yc,xo:xo+xc,0])

            elif i == 1:  # top right
                xo, yo = int(random.uniform(0, xc)), int(random.uniform(0, s-yc))
                img4[:yc,xc:,:]    = img[yo:yo+yc,xo:xo+(s-xc),:]
                labels4[:yc,xc:,0] = lab[yo:yo+yc,xo:xo+(s-xc),0]

                masks.append(lab[yo:yo+yc,xo:xo+(s-xc),0])

            elif i == 2:  # bottom left
                xo, yo = int(random.uniform(0, s-xc)), int(random.uniform(0, yc))
                img4[yc:,:xc,:]    = img[yo:yo+(s-yc),xo:xo+xc,:]
                labels4[yc:,:xc,0] = lab[yo:yo+(s-yc),xo:xo+xc,0]

                masks.append(lab[yo:yo+(s-yc),xo:xo+xc,0])

            elif i == 3:  # bottom right
                xo, yo = int(random.uniform(0, xc)), int(random.uniform(0, yc))
                img4[yc:,xc:,:]    = img[yo:yo+(s-yc),xo:xo+(s-xc),:]
                labels4[yc:,xc:,0] = lab[yo:yo+(s-yc),xo:xo+(s-xc),0]

                masks.append(lab[yo:yo+(s-yc),xo:xo+(s-xc),0])

        return img4, labels4


# --- PYTESTS --- #

def test_loader():
    """Test HDF5 dataloader with flips on and off."""
    run_batch(flip=False)
    run_batch(flip=True)


def run_batch(flip):
    """Sanity check for HDF5 dataloader checks for shapes and empty arrays."""
    # datasets to test loader on
    datasets = {
        'cell': (3, 256, 256),
        'mall': (3, 480, 640),
        'ucsd': (1, 160, 240)
    }

    # for each dataset check both training and validation HDF5
    # for each one check if shapes are right and arrays are not empty
    for dataset, size in datasets.items():
        for h5 in ('train.h5', 'valid.h5'):
            # create a loader in "all flips" or "no flips" mode
            data = H5Dataset(os.path.join(dataset, h5),
                             horizontal_flip=1.0 * flip,
                             vertical_flip=1.0 * flip)
            # create dataloader with few workers
            data_loader = DataLoader(data, batch_size=4, num_workers=4)

            # take one batch, check samples, and go to the next file
            for img, label in data_loader:
                # image batch shape (#workers, #channels, resolution)
                assert img.shape == (4, *size)
                # label batch shape (#workers, 1, resolution)
                assert label.shape == (4, 1, *size[1:])

                assert torch.sum(img) > 0
                assert torch.sum(label) > 0

                break
