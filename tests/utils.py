# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
import torch

from monai.utils.arrayutils import rescale_array

quick_test_var = "QUICKTEST"


def skip_if_quick(obj):
    is_quick = os.environ.get(quick_test_var, "").lower() == "true"

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)


def create_test_image(width, height, num_objs=12, rad_max=30, noise_max=0.0, num_seg_classes=5):
    """
    Return a noisy 2D image with `numObj' circles and a 2D mask image. The maximum radius of the circles is given as
    `radMax'. The mask will have `numSegClasses' number of classes for segmentations labeled sequentially from 1, plus a
    background class represented as 0. If `noiseMax' is greater than 0 then noise will be added to the image taken from
    the uniform distribution on range [0,noiseMax).
    """
    image = np.zeros((width, height))

    for i in range(num_objs):
        x = np.random.randint(rad_max, width - rad_max)
        y = np.random.randint(rad_max, height - rad_max)
        rad = np.random.randint(5, rad_max)
        spy, spx = np.ogrid[-x : width - x, -y : height - y]
        circle = (spx * spx + spy * spy) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(np.random.random() * num_seg_classes)
        else:
            image[circle] = np.random.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32)

    norm = np.random.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage = rescale_array(np.maximum(image, norm))

    return noisyimage, labels


class NumpyImageTestCase2D(unittest.TestCase):
    im_shape = (128, 128)
    input_channels = 1
    output_channels = 4
    num_classes = 3

    def setUp(self):
        im, msk = create_test_image(self.im_shape[0], self.im_shape[1], 4, 20, 0, self.num_classes)

        self.imt = im[None, None]
        self.seg1 = (msk[None, None] > 0).astype(np.float32)
        self.segn = msk[None, None]


class TorchImageTestCase2D(NumpyImageTestCase2D):

    def setUp(self):
        NumpyImageTestCase2D.setUp(self)
        self.imt = torch.tensor(self.imt)
        self.seg1 = torch.tensor(self.seg1)
        self.segn = torch.tensor(self.segn)
