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

from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union

import numpy as np

from monai.apps.utils import get_logger
from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data.utils import affine_to_spacing, ensure_tuple, orientation_ras_lps, to_affine_nd
from monai.transforms.spatial.array import SpatialResample
from monai.transforms.utils_pytorch_numpy_unification import ascontiguousarray, moveaxis
from monai.utils import GridSampleMode, GridSamplePadMode, convert_data_type, optional_import, require_pkg

DEFAULT_FMT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s"
logger = get_logger(module_name=__name__, fmt=DEFAULT_FMT)

if TYPE_CHECKING:
    import itk  # type: ignore
else:
    itk, _ = optional_import("itk", allow_namespace_pkg=True)

__all__ = ["ImageWriter", "ITKWriter", "logger"]


class ImageWriter:
    """
    The class is a collection of utilities to write images to disk.

    Main aspects to be considered are:

        - dimensionality of the data array, arrangements of spatial dimensions and channel/time dimensions
            - ``convert_to_channel_last()``
        - metadata of the current affine and output affine, the data array should be converted accordingly
            - ``get_meta_info()``
            - ``resample_if_needed()``
        - data type handling of the output image (as part of ``resample_if_needed()``)

    Subclasses of this class should implement the backend-specific functions:

        - ``set_data_array()`` to set the data array (input must be numpy array or torch tensor)
            - this method sets the backend object's data part
        - ``set_metadata()`` to set the metadata and output affine
            - this method sets the metadata including affine handling and image resampling
        - backend-specific data object ``create_backend_obj()``
        - backend-specific writing function ``write()``

    The primary usage of subclasses of ``ImageWriter`` is:

    .. code-block:: python

        writer = MyWriter()  # subclass of ImageWriter
        writer.set_data_array(data_array)
        writer.set_metadata(meta_dict)
        writer.write(filename)

    This creates an image writer object based on ``data_array`` and ``meta_dict`` and write to ``filename``.

    It supports up to three spatial dimensions (with the resampling step supports for both 2D and 3D).
    When saving multiple time steps or multiple channels `data_array`, time
    and/or modality axes should be the at the `channel_dim`. For example,
    the shape of a 2D eight-class and ``channel_dim=0``, the segmentation
    probabilities to be saved could be `(8, 64, 64)`; in this case
    ``data_array`` will be converted to `(64, 64, 1, 8)` (the third
    dimension is reserved as a spatial dimension).

    The ``metadata`` could optionally have the following keys:

        - ``'original_affine'``: for data original affine, it will be the
            affine of the output object, defaulting to an identity matrix.
        - ``'affine'``: it should specify the current data affine, defaulting to an identity matrix.
        - ``'spatial_shape'``: for data output spatial shape.

    When ``metadata`` is specified, the saver will may resample data from the space defined by
    `"affine"` to the space defined by `"original_affine"`, for more details, please refer to the
    ``resample_if_needed`` method.
    """

    def __init__(self, **kwargs):
        """
        The constructor supports adding new instance members.
        The current member in the base class is ``self.data_obj``, the subclasses can add more members,
        so that necessary meta information can be stored in the object and shared among the class methods.
        """
        self.data_obj = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_data_array(self, data_array, **kwargs):
        raise NotImplementedError(f"Subclasses of {self.__class__.__name__} must implement this method.")

    def set_metadata(self, meta_dict: Optional[Mapping], **options):
        raise NotImplementedError(f"Subclasses of {self.__class__.__name__} must implement this method.")

    def write(self, filename: PathLike, verbose: bool = True, **kwargs):
        """subclass should implement this method to call the backend-specific writing APIs."""
        if verbose:
            logger.info(f"writing: {filename}")

    @classmethod
    def create_backend_obj(cls, data_array: NdarrayOrTensor, **kwargs) -> np.ndarray:
        """
        Subclass should implement this method to return a backend-specific data representation object.
        This method is used by ``cls.write`` and the input ``data_array`` is assumed 'channel-last'.
        """
        return convert_data_type(data_array, np.ndarray)[0]  # type: ignore

    @classmethod
    def resample_if_needed(
        cls,
        data_array: NdarrayOrTensor,
        affine: Optional[NdarrayOrTensor] = None,
        target_affine: Optional[NdarrayOrTensor] = None,
        output_spatial_shape: Union[Sequence[int], int, None] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ):
        """
        Convert the ``data_array`` into the coordinate system specified by
        ``target_affine``, from the current coordinate definition of ``affine``.

        If the transform between ``affine`` and ``target_affine`` could be
        achieved by simply transposing and flipping ``data_array``, no resampling
        will happen.  Otherwise, this function resamples ``data_array`` using the
        transformation computed from ``affine`` and ``target_affine``.

        This function assumes the NIfTI dimension notations. Spatially it
        supports up to three dimensions, that is, H, HW, HWD for 1D, 2D, 3D
        respectively. When saving multiple time steps or multiple channels,
        time and/or modality axes should be appended after the first three
        dimensions. For example, shape of 2D eight-class segmentation
        probabilities to be saved could be `(64, 64, 1, 8)`. Also, data in
        shape `(64, 64, 8)` or `(64, 64, 8, 1)` will be considered as a
        single-channel 3D image. The ``convert_to_channel_last`` method can be
        used to convert the data to the format described here.

        Note that the shape of the resampled ``data_array`` may subject to some
        rounding errors. For example, resampling a 20x20 pixel image from pixel
        size (1.5, 1.5)-mm to (3.0, 3.0)-mm space will return a 10x10-pixel
        image. However, resampling a 20x20-pixel image from pixel size (2.0,
        2.0)-mm to (3.0, 3.0)-mm space will output a 14x14-pixel image, where
        the image shape is rounded from 13.333x13.333 pixels. In this case
        ``output_spatial_shape`` could be specified so that this function
        writes image data to a designated shape.

        Args:
            data_array: input data array to be converted.
            affine: the current affine of ``data_array``. Defaults to identity
            target_affine: the designated affine of ``data_array``.
                The actual output affine might be different from this value due to precision changes.
            output_spatial_shape: spatial shape of the output image.
                This option is used when resampling is needed.
            mode: available options are {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
                This option is used when resampling is needed.
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: available options are {``"zeros"``, ``"border"``, ``"reflection"``}.
                This option is used when resampling is needed.
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: boolean option of ``grid_sample`` to handle the corner convention.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to
                ``np.float64`` for best precision. If ``None``, use the data type of input data.
                The output data type of this method is always ``np.float32``.
        """
        resampler = SpatialResample(mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype)
        output_array, target_affine = resampler(
            data_array[None], src_affine=affine, dst_affine=target_affine, spatial_size=output_spatial_shape
        )
        return output_array[0], target_affine

    @classmethod
    def convert_to_channel_last(
        cls,
        data: NdarrayOrTensor,
        channel_dim: Union[None, int, Sequence[int]] = 0,
        squeeze_end_dims: bool = True,
        spatial_ndim: Optional[int] = 3,
        contiguous: bool = False,
    ):
        """
        Rearrange the data array axes to make the `channel_dim`-th dim the last
        dimension and ensure there are ``spatial_ndim`` number of spatial
        dimensions.

        When ``squeeze_end_dims`` is ``True``, a postprocessing step will be
        applied to remove any trailing singleton dimensions.

        Args:
            data: input data to be converted to "channel-last" format.
            channel_dim: specifies the channel axes of the data array to move to the last.
                ``None`` indicates no channel dimension, a new axis will be appended as the channel dimension.
                a sequence of integers indicates multiple non-spatial dimensions.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is `(H,W,D,C)` and C==1, then it will be saved as `(H,W,D)`.
                If D is also 1, it will be saved as `(H,W)`. If ``False``, image will always be saved as `(H,W,D,C)`.
            spatial_ndim: modifying the spatial dims if needed, so that output to have at least
                this number of spatial dims. If ``None``, the output will have the same number of
                spatial dimensions as the input.
            contiguous: if ``True``, the output will be contiguous.
        """
        # change data to "channel last" format
        if channel_dim is not None:
            _chns = ensure_tuple(channel_dim)
            data = moveaxis(data, _chns, tuple(range(-len(_chns), 0)))
        else:  # adds a channel dimension
            data = data[..., None]
        # To ensure at least ``spatial_ndim`` number of spatial dims
        if spatial_ndim:
            while len(data.shape) < spatial_ndim + 1:  # assuming the data has spatial + channel dims
                data = data[..., None, :]
            while len(data.shape) > spatial_ndim + 1:
                data = data[..., 0, :]
        # if desired, remove trailing singleton dimensions
        while squeeze_end_dims and data.shape[-1] == 1:
            data = np.squeeze(data, -1)
        if contiguous:
            data = ascontiguousarray(data)
        return data

    @classmethod
    def get_meta_info(cls, metadata: Optional[Mapping] = None):
        """
        Extracts relevant meta information from the metadata object (using ``.get``).
        Optional keys are ``"spatial_shape"``, ``"affine"``, ``"original_affine"``.
        """
        if not metadata:
            metadata = {"original_affine": None, "affine": None, "spatial_shape": None}
        original_affine = metadata.get("original_affine")
        affine = metadata.get("affine")
        spatial_shape = metadata.get("spatial_shape")
        return original_affine, affine, spatial_shape


@require_pkg(pkg_name="itk")
class ITKWriter(ImageWriter):
    """
    Write data and metadata into files on disk using ITK-python.

    .. code-block:: python

        import numpy as np
        from monai.data import ITKWriter

        np_data = np.arange(48).reshape(3, 4, 4)

        # write as 3d spatial image no channel
        writer = ITKWriter(output_dtype=np.float32)
        writer.set_data_array(np_data, channel_dim=None)
        # optionally set metadata affine
        writer.set_metadata({"affine": np.eye(4), "original_affine": -1 * np.eye(4)})
        writer.write("test1.nii.gz")

        # write as 2d image, channel-first
        writer = ITKWriter(output_dtype=np.uint8)
        writer.set_data_array(np_data, channel_dim=0)
        writer.set_metadata({"spatial_shape": (5, 5)})
        writer.write("test1.png")

    """

    def __init__(self, output_dtype: DtypeLike = np.float32, **kwargs):
        """
        Args:
            output_dtype: output data type.
            kwargs: keyword arguments passed to ``ImageWriter``.

        The constructor will create ``self.output_dtype`` internally.
        ``affine`` and ``channel_dim`` are initialized as instance members (default ``None``, ``0``):

            - user-specified ``affine`` should be set in ``set_metadata``,
            - user-specified ``channel_dim`` should be set in ``set_data_array``.
        """
        super().__init__(output_dtype=output_dtype, affine=None, channel_dim=0, **kwargs)

    def set_data_array(self, data_array, channel_dim: Optional[int] = 0, squeeze_end_dims: bool = True, **kwargs):
        """
        Convert ``data_array`` into 'channel-last' numpy ndarray.

        Args:
            data_array: input data array with the channel dimension specified by ``channel_dim``.
            channel_dim: channel dimension of the data array. Defaults to 0.
                ``None`` indicates data without any channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed.
            kwargs: keyword arguments passed to ``self.convert_to_channel_last``,
                currently support ``spatial_ndim`` and ``contiguous``, defauting to ``3`` and ``False`` respectively.
        """
        self.data_obj = self.convert_to_channel_last(
            data=data_array,
            channel_dim=channel_dim,
            squeeze_end_dims=squeeze_end_dims,
            spatial_ndim=kwargs.pop("spatial_ndim", 3),
            contiguous=kwargs.pop("contiguous", True),
        )
        self.channel_dim = channel_dim

    def set_metadata(self, meta_dict: Optional[Mapping] = None, resample: bool = True, **options):
        """
        Resample ``self.dataobj`` if needed.  This method assumes ``self.data_obj`` is a 'channel-last' ndarray.

        Args:
            meta_dict: a metadata dictionary for affine, original affine and spatial shape information.
                Optional keys are ``"spatial_shape"``, ``"affine"``, ``"original_affine"``.
            resample: if ``True``, the data will be resampled to the original affine (specified in ``meta_dict``).
            options: keyword arguments passed to ``self.resample_if_needed``,
                currently support ``mode``, ``padding_mode``, ``align_corners``, and ``dtype``,
                defaulting to ``bilinear``, ``border``, ``False``, and ``np.float64`` respectively.
        """
        original_affine, affine, spatial_shape = self.get_meta_info(meta_dict)
        self.data_obj, self.affine = self.resample_if_needed(
            data_array=self.data_obj,
            affine=affine,
            target_affine=original_affine if resample else None,
            output_spatial_shape=spatial_shape,
            mode=options.pop("mode", GridSampleMode.BILINEAR),
            padding_mode=options.pop("padding_mode", GridSamplePadMode.BORDER),
            align_corners=options.pop("align_corners", False),
            dtype=options.pop("dtype", np.float64),
        )

    def write(self, filename: PathLike, verbose: bool = False, **kwargs):
        """
        Create an ITK object from ``self.create_backend_obj(self.obj, ...)`` and call ``itk.imwrite``.

        Args:
            filename: filename or PathLike object.
            verbose: if ``True``, log the progress.
            kwargs: keyword arguments passed to ``itk.imwrite``,
                currently support ``compression`` and ``imageio``.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Wrapping/Generators/Python/itk/support/extras.py#L809
        """
        super().write(filename, verbose=verbose)
        self.data_obj = self.create_backend_obj(
            self.data_obj, channel_dim=self.channel_dim, affine=self.affine, dtype=self.output_dtype, **kwargs  # type: ignore
        )
        itk.imwrite(
            self.data_obj, filename, compression=kwargs.pop("compression", False), imageio=kwargs.pop("imageio", None)
        )

    @classmethod
    def create_backend_obj(
        cls,
        data_array: NdarrayOrTensor,
        channel_dim: Optional[int] = 0,
        affine: Optional[NdarrayOrTensor] = None,
        dtype: DtypeLike = np.float32,
        **kwargs,
    ):
        """
        Create an ITK object from ``data_array``. This method assumes a 'channel-last' ``data_array``.

        Args:
            data_array: input data array.
            channel_dim: channel dimension of the data array. This is used to create a Vector Image if it is not ``None``.
            affine: affine matrix of the data array. This is used to compute `spacing`, `direction` and `origin`.
            dtype: output data type.
            kwargs: keyword arguments. Current `itk.GetImageFromArray` will read ``ttype`` from this dictionary.

        see also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Wrapping/Generators/Python/itk/support/extras.py#L389
        """
        data_array = super().create_backend_obj(data_array)
        _is_vec = channel_dim is not None
        if _is_vec:
            data_array = np.moveaxis(data_array, -1, 0)  # from channel last to channel first
        data_array = data_array.T.astype(dtype, copy=True, order="C")
        itk_obj = itk.GetImageFromArray(data_array, is_vector=_is_vec, ttype=kwargs.pop("ttype", None))

        d = len(itk.size(itk_obj))
        if affine is None:
            affine = np.eye(d + 1, dtype=np.float64)
        _affine = convert_data_type(affine, np.ndarray)[0]
        _affine = orientation_ras_lps(to_affine_nd(d, _affine))
        spacing = affine_to_spacing(_affine, r=d)
        _direction: np.ndarray = np.diag(1 / spacing)
        _direction = _affine[:d, :d] @ _direction  # type: ignore
        itk_obj.SetSpacing(spacing.tolist())
        itk_obj.SetOrigin(_affine[:d, -1].tolist())
        itk_obj.SetDirection(itk.GetMatrixFromArray(_direction))
        return itk_obj
