# Copyright (c) MONAI Consortium
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
A collection of "vanilla" transforms for box operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional, Sequence, Type, Union

import numpy as np

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import BoxMode, convert_box_mode, convert_box_to_standard_mode, get_spatial_dims
from monai.transforms.transform import Transform
from monai.utils import ensure_tuple, ensure_tuple_rep, fall_back_tuple, look_up_option
from monai.utils.enums import TransformBackends

from .box_ops import apply_affine_to_boxes, flip_boxes, resize_boxes, zoom_boxes

__all__ = ["ConvertBoxToStandardMode", "ConvertBoxMode", "AffineBox", "ZoomBox", "ResizeBox", "FlipBox"]


class ConvertBoxMode(Transform):
    """
    This transform converts the boxes in src_mode to the dst_mode.

    Args:
        src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
        dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.

    Note:
        ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
        also represented as "xyxy" for 2D and "xyzxyz" for 3D.

        src_mode and dst_mode can be:
            #. str: choose from :class:`~monai.utils.enums.BoxModeName`, for example,
                - "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                - "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                - "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                - "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                - "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                - "xywh": boxes has format [xmin, ymin, xsize, ysize]
                - "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                - "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                - "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
            #. None: will assume mode is ``StandardMode()``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,4)
            # convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_converter = ConvertBoxMode(src_mode="xyxy", dst_mode="ccwh")
            box_converter(boxes)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
    ) -> None:
        self.src_mode = src_mode
        self.dst_mode = dst_mode

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Converts the boxes in src_mode to the dst_mode.

        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

        Returns:
            bounding boxes with target mode, with same data type as ``boxes``, does not share memory with ``boxes``
        """
        return convert_box_mode(boxes, src_mode=self.src_mode, dst_mode=self.dst_mode)


class ConvertBoxToStandardMode(Transform):
    """
    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Args:
        mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_converter = ConvertBoxToStandardMode(mode="xxyyzz")
            box_converter(boxes)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode: Union[str, BoxMode, Type[BoxMode], None] = None) -> None:
        self.mode = mode

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Convert given boxes to standard mode.
        Standard mode is "xyxy" or "xyzxyz",
        representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

        Returns:
            bounding boxes with standard mode, with same data type as ``boxes``, does not share memory with ``boxes``
        """
        return convert_box_to_standard_mode(boxes, mode=self.mode)


class AffineBox(Transform):
    """
    Applies affine matrix to the boxes
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, boxes: NdarrayOrTensor, affine: Union[NdarrayOrTensor, None]) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            affine: affine matrix to be applied to the box coordinate
        """
        if affine is None:
            return boxes

        return apply_affine_to_boxes(boxes, affine=affine)


class ZoomBox(Transform):
    """
    Zooms an ND Box with same padding or slicing setting with Zoom().

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, zoom: Union[Sequence[float], float], keep_size: bool = False, **kwargs) -> None:
        self.zoom = zoom
        self.keep_size = keep_size
        self.kwargs = kwargs

    def __call__(
        self, boxes: NdarrayOrTensor, src_spatial_size: Union[Sequence[int], int, None] = None
    ) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            src_spatial_size: original image spatial size before zooming, used only when keep_size=True.
        """
        spatial_dims: int = get_spatial_dims(boxes=boxes)
        self._zoom = ensure_tuple_rep(self.zoom, spatial_dims)  # match the spatial image dim

        if not self.keep_size:
            return zoom_boxes(boxes, self._zoom)

        if src_spatial_size is None:
            raise ValueError("keep_size=True, src_spatial_size must be provided.")

        src_spatial_size = ensure_tuple_rep(src_spatial_size, spatial_dims)
        dst_spatial_size = [int(round(z * ss)) for z, ss in zip(self._zoom, src_spatial_size)]
        self._zoom = tuple(ds / float(ss) for ss, ds in zip(src_spatial_size, dst_spatial_size))
        zoomed_boxes = zoom_boxes(boxes, self._zoom)

        # See also keep_size in monai.transforms.spatial.array.Zoom()
        if not np.allclose(np.array(src_spatial_size), np.array(dst_spatial_size)):
            for axis, (od, zd) in enumerate(zip(src_spatial_size, dst_spatial_size)):
                diff = od - zd
                half = abs(diff) // 2
                if diff > 0:  # need padding (half, diff - half)
                    zoomed_boxes[:, axis] = zoomed_boxes[:, axis] + half
                    zoomed_boxes[:, axis + spatial_dims] = zoomed_boxes[:, axis + spatial_dims] + half
                elif diff < 0:  # need slicing (half, half + od)
                    zoomed_boxes[:, axis] = zoomed_boxes[:, axis] - half
                    zoomed_boxes[:, axis + spatial_dims] = zoomed_boxes[:, axis + spatial_dims] - half
        return zoomed_boxes


class ResizeBox(Transform):
    """
    Resize the input boxes when the corresponding image is
    resized to given spatial size (with scaling, not cropping/padding).

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_size: Union[Sequence[int], int], size_mode: str = "all", **kwargs) -> None:
        self.size_mode = look_up_option(size_mode, ["all", "longest"])
        self.spatial_size = spatial_size

    def __call__(  # type: ignore
        self, boxes: NdarrayOrTensor, src_spatial_size: Union[Sequence[int], int]
    ) -> NdarrayOrTensor:
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            src_spatial_size: original image spatial size before resizing.

        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``boxes`` spatial dimensions.
        """
        input_ndim = get_spatial_dims(boxes=boxes)  # spatial ndim
        src_spatial_size_ = ensure_tuple_rep(src_spatial_size, input_ndim)

        if self.size_mode == "all":
            # spatial_size must be a Sequence if size_mode is 'all'
            output_ndim = len(ensure_tuple(self.spatial_size))
            if output_ndim != input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            spatial_size_ = fall_back_tuple(self.spatial_size, src_spatial_size_)
        else:  # for the "longest" mode
            if not isinstance(self.spatial_size, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = self.spatial_size / max(src_spatial_size_)
            spatial_size_ = tuple(int(round(s * scale)) for s in src_spatial_size_)

        return resize_boxes(boxes, src_spatial_size_, spatial_size_)


class FlipBox(Transform):
    """
    Reverses the box coordinates along the given spatial axis. Preserves shape.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        self.spatial_axis = spatial_axis

    def __call__(  # type: ignore
        self, boxes: NdarrayOrTensor, spatial_size: Union[Sequence[int], int]
    ) -> NdarrayOrTensor:
        """
        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            spatial_size: image spatial size.
        """

        return flip_boxes(boxes, spatial_size=spatial_size, flip_axes=self.spatial_axis)
