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
"""
A collection of "vanilla" transforms
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import numpy as np
import torch
from skimage.transform import resize
import scipy.ndimage

import monai
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.compose import Randomizable
from monai.transforms.utils import rescale_array

export = monai.utils.export("monai.transforms")


@export
class AddChannel:
    """
    Adds a 1-length channel dimension to the input image.
    """

    def __call__(self, img):
        return img[None]


@export
class Transpose:
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices):
        self.indices = indices

    def __call__(self, img):
        return img.transpose(self.indices)


@export
class Rescale:
    """
    Rescales the input image to the given value range.
    """

    def __init__(self, minv=0.0, maxv=1.0, dtype=np.float32):
        self.minv = minv
        self.maxv = maxv
        self.dtype = dtype

    def __call__(self, img):
        return rescale_array(img, self.minv, self.maxv, self.dtype)


@export
class GaussianNoise(Randomizable):
    """Add gaussian noise to image.

    Args:
        mean (float or array of floats): Mean or “centre” of the distribution.
        scale (float): Standard deviation (spread) of distribution.
        size (int or tuple of ints): Output shape. Default: None (single value is returned).
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return img + self.R.normal(self.mean, self.R.uniform(0, self.std), size=img.shape)


@export
class Flip:
    """Reverses the order of elements along the given axis. Preserves shape.
    Uses np.flip in practice. See numpy.flip for additional details.

    Args:
        axes (None, int or tuple of ints): Axes along which to flip over. Default is None.
    """

    def __init__(self, axis=None):
        assert axis is None or isinstance(axis, (int, list, tuple)), \
            "axis must be None, int or tuple of ints."
        self.axis = axis

    def __call__(self, img):
        return np.flip(img, self.axis)


@export
class Resize:
    """
    Resize the input image to given resolution. Uses skimage.transform.resize underneath.
    For additional details, see https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize.

    Args:
        order (int): Order of spline interpolation. Default=1.
        mode (str): Points outside boundaries are filled according to given mode. 
            Options are 'constant', 'edge', 'symmetric', 'reflect', 'wrap'.
        cval (float): Used with mode 'constant', the value outside image boundaries.
        clip (bool): Wheter to clip range of output values after interpolation. Default: True.
        preserve_range (bool): Whether to keep original range of values. Default is True.
            If False, input is converted according to conventions of img_as_float. See 
            https://scikit-image.org/docs/dev/user_guide/data_types.html.
        anti_aliasing (bool): Whether to apply a gaussian filter to image before down-scaling.
            Default is True.
        anti_aliasing_sigma (float, tuple of floats): Standard deviation for gaussian filtering.
    """

    def __init__(self, output_shape, order=1, mode='reflect', cval=0,
                 clip=True, preserve_range=True, 
                 anti_aliasing=True, anti_aliasing_sigma=None):
        assert isinstance(order, int), "order must be integer."
        self.output_shape = output_shape
        self.order = order
        self.mode = mode
        self.cval = cval
        self.clip = clip
        self.preserve_range = preserve_range
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma

    def __call__(self, img):
        return resize(img, self.output_shape, order=self.order,
                      mode=self.mode, cval=self.cval,
                      clip=self.clip, preserve_range=self.preserve_range,
                      anti_aliasing=self.anti_aliasing, 
                      anti_aliasing_sigma=self.anti_aliasing_sigma)


@export
class Rotate:
    """
    Rotates an input image by given angle. Uses scipy.ndimage.rotate. For more details, see
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.ndimage.rotate.html.

    Args:
        angle (float): Rotation angle in degrees.
        axes (tuple of 2 ints): Axes of rotation. Default: (1, 2). This is the first two
            axis in spatial dimensions according to MONAI channel first shape assumption.
        reshape (bool): If true, output shape is made same as input. Default: True.
        order (int): Order of spline interpolation. Range 0-5. Default: 1. This is
            different from scipy where default interpolation is 3.
        mode (str): Points outside boundary filled according to this mode. Options are 
            'constant', 'nearest', 'reflect', 'wrap'. Default: 'constant'.
        cval (scalar): Values to fill outside boundary. Default: 0.
        prefiter (bool): Apply spline_filter before interpolation. Default: True.
    """

    def __init__(self, angle, axes=(1, 2), reshape=True, order=1, 
                 mode='constant', cval=0, prefilter=True):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.axes = axes

    def __call__(self, img):
        return scipy.ndimage.rotate(img, self.angle, self.axes,
                                    reshape=self.reshape, order=self.order, 
                                    mode=self.mode, cval=self.cval, 
                                    prefilter=self.prefilter)


@export
class Zoom:
    """ Zooms a nd image. Uses scipy.ndimage.zoom or cupyx.scipy.ndimage.zoom in case of gpu. 
    For details, please see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html.

    Args:
        zoom (float or sequence): The zoom factor along the axes. If a float, zoom is the same for each axis. 
            If a sequence, zoom should contain one value for each axis.
        order (int): order of interpolation. Default=3.
        mode (str): Determines how input is extended beyond boundaries. Default is 'constant'.
        cval (scalar, optional): Value to fill past edges. Default is 0.
        use_gpu (bool): Should use cpu or gpu. Uses cupyx which doesn't support order > 1 and modes
            'wrap' and 'reflect'. Defaults to cpu for these cases or if cupyx not found.
        keep_size (bool): Should keep original size (pad if needed).
    """
    def __init__(self, zoom, order=3, mode='constant', cval=0, prefilter=True, use_gpu=False, keep_size=False):
        assert isinstance(order, int), "Order must be integer."
        self.zoom = zoom
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.use_gpu = use_gpu
        self.keep_size = keep_size

    def __call__(self, img):
        zoomed = None
        if self.use_gpu:
            try:
                import cupy
                from cupyx.scipy.ndimage import zoom as zoom_gpu

                zoomed_gpu = zoom_gpu(cupy.array(img), zoom=self.zoom, order=self.order,
                                      mode=self.mode, cval=self.cval, prefilter=self.prefilter)
                zoomed = cupy.asnumpy(zoomed_gpu)
            except ModuleNotFoundError:
                print('For GPU zoom, please install cupy. Defaulting to cpu.')
            except NotImplementedError:
                print("Defaulting to CPU. cupyx doesn't support order > 1 and modes 'wrap' or 'reflect'.")

        if zoomed is None:
            zoomed = scipy.ndimage.zoom(img, zoom=self.zoom, order=self.order,
                                        mode=self.mode, cval=self.cval, prefilter=self.prefilter)

        # Crops to original size or pads.
        if self.keep_size:
            shape = img.shape
            pad_vec = [[0, 0]] * len(shape)
            crop_vec = list(zoomed.shape)
            for d in range(len(shape)):
                if zoomed.shape[d] > shape[d]:
                    crop_vec[d] = shape[d]
                elif zoomed.shape[d] < shape[d]:
                    # pad_vec[d] = [0, shape[d] - zoomed.shape[d]]
                    pad_h = (float(shape[d]) - float(zoomed.shape[d])) / 2
                    pad_vec[d] = [int(np.floor(pad_h)), int(np.ceil(pad_h))]
            zoomed = zoomed[0:crop_vec[0], 0:crop_vec[1], 0:crop_vec[2]]
            zoomed = np.pad(zoomed, pad_vec, mode='constant', constant_values=self.cval)

        return zoomed


@export
class ToTensor:
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img):
        return torch.from_numpy(img)


@export
class UniformRandomPatch(Randomizable):
    """
    Selects a patch of the given size chosen at a uniformly random position in the image.
    """

    def __init__(self, patch_size):
        self.patch_size = (None,) + tuple(patch_size)

        self._slices = None

    def randomize(self, image_shape, patch_shape):
        self._slices = get_random_patch(image_shape, patch_shape, self.R)

    def __call__(self, img):
        patch_size = get_valid_patch_size(img.shape, self.patch_size)
        self.randomize(img.shape, patch_size)
        return img[self._slices]


@export
class IntensityNormalizer:
    """Normalize input based on provided args, using calculated mean and std if not provided
    (shape of subtrahend and divisor must match. if 0, entire volume uses same subtrahend and
     divisor, otherwise the shape can have dimension 1 for channels).
     Current implementation can only support 'channel_last' format data.

    Args:
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
        dtype: output data format
    """

    def __init__(self, subtrahend=None, divisor=None, dtype=np.float32):
        if subtrahend is not None or divisor is not None:
            assert isinstance(subtrahend, np.ndarray) and isinstance(divisor, np.ndarray), \
                'subtrahend and divisor must be set in pair and in numpy array.'
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.dtype = dtype

    def __call__(self, img):
        if self.subtrahend is not None and self.divisor is not None:
            img -= self.subtrahend
            img /= self.divisor
        else:
            img -= np.mean(img)
            img /= np.std(img)

        if self.dtype != img.dtype:
            img = img.astype(self.dtype)
        return img


@export
class ImageEndPadder:
    """Performs padding by appending to the end of the data all on one side for each dimension.
     Uses np.pad so in practice, a mode needs to be provided. See numpy.lib.arraypad.pad
     for additional details.

    Args:
        out_size (list): the size of region of interest at the end of the operation.
        mode (string): a portion from numpy.lib.arraypad.pad is copied below.
        dtype: output data format.
    """

    def __init__(self, out_size, mode, dtype=np.float32):
        assert out_size is not None and isinstance(out_size, (list, tuple)), 'out_size must be list or tuple'
        self.out_size = out_size
        assert isinstance(mode, str), 'mode must be str'
        self.mode = mode
        self.dtype = dtype

    def _determine_data_pad_width(self, data_shape):
        return [(0, max(self.out_size[i] - data_shape[i], 0)) for i in range(len(self.out_size))]

    def __call__(self, img):
        data_pad_width = self._determine_data_pad_width(img.shape[2:])
        all_pad_width = [(0, 0), (0, 0)] + data_pad_width
        img = np.pad(img, all_pad_width, self.mode)
        return img


@export
class Rotate90:
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k=1, axes=(1, 2)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            axes (2 ints): defines the plane to rotate with 2 axes.
        """
        self.k = k
        self.plane_axes = axes

    def __call__(self, img):
        return np.ascontiguousarray(np.rot90(img, self.k, self.plane_axes))


@export
class RandRotate90(Randomizable):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `axes`.
    """

    def __init__(self, prob=0.1, max_k=3, axes=(1, 2)):
        """
        Args:
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            axes (2 ints): defines the plane to rotate with 2 axes.
                (Default (1, 2))
        """
        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.axes = axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self):
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate90(self._rand_k, self.axes)
        return rotator(img)


@export
class SpatialCrop:
    """General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop 1, 2 or 3 dimensions spatial data.
    Either a center and size must be provided, or alternatively if center and size
    are not provided, the start and end coordinates of the ROI must be provided.
    The sub-volume must sit the within original image.

    Note: This transform will not work if the crop region is larger than the image itself.
    """

    def __init__(self, roi_center=None, roi_size=None, roi_start=None, roi_end=None):
        """
        Args:
            roi_center (list or tuple): voxel coordinates for center of the crop ROI.
            roi_size (list or tuple): size of the crop ROI.
            roi_start (list or tuple): voxel coordinates for start of the crop ROI.
            roi_end (list or tuple): voxel coordinates for end of the crop ROI.
        """
        if roi_center is not None and roi_size is not None:
            assert isinstance(roi_center, (list, tuple)), 'roi_center must be list or tuple.'
            assert isinstance(roi_size, (list, tuple)), 'roi_size must be list or tuple.'
            assert all(x > 0 for x in roi_center), 'all elements of roi_center must be positive.'
            assert all(x > 0 for x in roi_size), 'all elements of roi_size must be positive.'
            roi_center = np.asarray(roi_center, dtype=np.uint16)
            roi_size = np.asarray(roi_size, dtype=np.uint16)
            self.roi_start = np.subtract(roi_center, np.floor_divide(roi_size, 2))
            self.roi_end = np.add(self.roi_start, roi_size)
        else:
            assert roi_start is not None and roi_end is not None, 'roi_start and roi_end must be provided.'
            assert isinstance(roi_start, (list, tuple)), 'roi_start must be list or tuple.'
            assert isinstance(roi_end, (list, tuple)), 'roi_end must be list or tuple.'
            assert all(x >= 0 for x in roi_start), 'all elements of roi_start must be greater than or equal to 0.'
            assert all(x > 0 for x in roi_end), 'all elements of roi_end must be positive.'
            self.roi_start = roi_start
            self.roi_end = roi_end

    def __call__(self, img):
        max_end = img.shape[1:]
        assert (np.subtract(max_end, self.roi_start) >= 0).all(), 'roi start out of image space.'
        assert (np.subtract(max_end, self.roi_end) >= 0).all(), 'roi end out of image space.'
        assert (np.subtract(self.roi_end, self.roi_start) >= 0).all(), 'invalid roi range.'
        if len(self.roi_start) == 1:
            data = img[:, self.roi_start[0]:self.roi_end[0]].copy()
        elif len(self.roi_start) == 2:
            data = img[:, self.roi_start[0]:self.roi_end[0], self.roi_start[1]:self.roi_end[1]].copy()
        elif len(self.roi_start) == 3:
            data = img[:, self.roi_start[0]:self.roi_end[0], self.roi_start[1]:self.roi_end[1],
                       self.roi_start[2]:self.roi_end[2]].copy()
        else:
            raise ValueError('unsupported image shape.')
        return data


@export
class RandZoom(Randomizable):
    """Randomly zooms input arrays with given probability within given zoom range.

    Args:
        prob (float): Probability of zooming.
        min_zoom (float or sequence): Min zoom factor. Can be float or sequence same size as image.
        max_zoom (float or sequence): Max zoom factor. Can be float or sequence same size as image.
        order (int): order of interpolation. Default=3.
        mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap'): Determines how input is 
            extended beyond boundaries. Default: 'constant'.
        cval (scalar, optional): Value to fill past edges. Default is 0.
        use_gpu (bool): Should use cpu or gpu. Uses cupyx which doesn't support order > 1 and modes
            'wrap' and 'reflect'. Defaults to cpu for these cases or if cupyx not found.
        keep_size (bool): Should keep original size (pad if needed).
    """

    def __init__(self, prob=0.1, min_zoom=0.9, max_zoom=1.1, order=3,
                 mode='constant', cval=0, prefilter=True,
                 use_gpu=False, keep_size=False):
        if hasattr(min_zoom, '__iter__') and \
           hasattr(max_zoom, '__iter__'):
            assert len(min_zoom) == len(max_zoom), "min_zoom and max_zoom must have same length."
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.prob = prob
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.use_gpu = use_gpu
        self.keep_size = keep_size

        self._do_transform = False
        self._zoom = None

    def randomize(self):
        self._do_transform = self.R.random_sample() < self.prob
        if hasattr(self.min_zoom, '__iter__'):
            self._zoom = (self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom))
        else:
            self._zoom = self.R.uniform(self.min_zoom, self.max_zoom)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        zoomer = Zoom(self._zoom, self.order, self.mode, self.cval, self.prefilter, self.use_gpu, self.keep_size)
        return zoomer(img)
