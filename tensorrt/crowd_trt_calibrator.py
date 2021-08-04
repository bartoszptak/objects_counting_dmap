"""crowd_trt_calibrator.py

The original code could be found in TensorRT-7.x sample code:
"samples/python/int8_caffe_mnist/calibrator.py". The modifications
allow to handle with 5-channels input (R,G,B,Fx,Fy) which contains 
3-channels from RGB image and 2-channels optical flow. Due to usage 
of several sequences, corresponding optical flow data should be 
prepared before calibration. Structure of calibration data directory:

.
├── imgs
└── imgs_flows

In order to prepare data one can use prepare_calibration_data.py script.
"""

#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.


import os
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from glob import glob


def _warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res


def _preprocess_crowd(img, flow_img, input_shape=(608,608)):
    """Preprocess an image before TRT crowd inferencing.

    # Args
        img: uint8 numpy array of shape either (img_h, img_w, 3)
             or (img_h, img_w)
        input_shape: a tuple of (H, W)

    # Returns
        preprocessed img: float32 numpy array of shape (5, H, W)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.concatenate((img, np.reshape(flow_img*255., (*img.shape[:2], 2))), axis=2)

    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = img.astype(np.float32) / 255.0
    return img.transpose((2, 0, 1)).astype(np.float32)


class CrowdEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """CrowdEntropyCalibrator

    This class implements TensorRT's IInt8EntropyCalibtrator2 interface.
    It reads all images from the specified directory and generates INT8
    calibration data for crowd models accordingly.
    """

    def __init__(self, img_dir, net_hw, cache_file, batch_size=1):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError('%s does not exist' % img_dir)

        super().__init__()  # trt.IInt8EntropyCalibrator2.__init__(self)

        self.img_dir = img_dir
        self.net_hw = net_hw
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.blob_size = 5 * net_hw[0] * net_hw[1] * np.dtype('float32').itemsize * batch_size

        self.inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.inst.setUseSpatialPropagation(False)

        self.jpgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # The number "500" is NVIDIA's suggestion.  See here:
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c
        if len(self.jpgs) < 500:
            print('WARNING: found less than 500 images in %s!' % img_dir)
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.blob_size)

    def __del__(self):
        del self.device_input  # free CUDA memory

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.jpgs):
            return None

        batch = []
        for i in range(self.batch_size):
            img_path = os.path.join(
                self.img_dir, self.jpgs[self.current_index + i])
            flow_path = img_path.replace('imgs', 'imgs_flows').replace('jpg', 'npy')
            img = cv2.imread(img_path)
            assert img is not None, 'failed to read %s' % img_path

            flow_img = np.load(flow_path)
            assert flow_img is not None, 'failed to read %s' % flow_path

            img = _preprocess_crowd(img, flow_img)

            batch.append(img)
        batch = np.stack(batch)
        assert batch.nbytes == self.blob_size

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
