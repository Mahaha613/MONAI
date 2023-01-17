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

from __future__ import annotations

import torch

from monai.utils.module import optional_import

_C, _ = optional_import("monai._C")

__all__ = ["BilateralFilter", "PHLFilter", "TrainableBilateralFilter"]


class BilateralFilter(torch.autograd.Function):
    """
    Blurs the input tensor spatially whilst preserving edges. Can run on 1D, 2D, or 3D,
    tensors (on top of Batch and Channel dimensions). Two implementations are provided,
    an exact solution and a much faster approximation which uses a permutohedral lattice.

    See:
        https://en.wikipedia.org/wiki/Bilateral_filter
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor.

        spatial sigma: the standard deviation of the spatial blur. Higher values can
            hurt performance when not using the approximate method (see fast approx).

        color sigma: the standard deviation of the color blur. Lower values preserve
            edges better whilst higher values tend to a simple gaussian spatial blur.

        fast approx: This flag chooses between two implementations. The approximate method may
            produce artifacts in some scenarios whereas the exact solution may be intolerably
            slow for high spatial standard deviations.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, spatial_sigma=5, color_sigma=0.5, fast_approx=True):
        ctx.ss = spatial_sigma
        ctx.cs = color_sigma
        ctx.fa = fast_approx
        output_data = _C.bilateral_filter(input, spatial_sigma, color_sigma, fast_approx)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        spatial_sigma, color_sigma, fast_approx = ctx.ss, ctx.cs, ctx.fa
        grad_input = _C.bilateral_filter(grad_output, spatial_sigma, color_sigma, fast_approx)
        return grad_input, None, None, None


class PHLFilter(torch.autograd.Function):
    """
    Filters input based on arbitrary feature vectors. Uses a permutohedral
    lattice data structure to efficiently approximate n-dimensional gaussian
    filtering. Complexity is broadly independent of kernel size. Most applicable
    to higher filter dimensions and larger kernel sizes.

    See:
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor to be filtered.

        features: feature tensor used to filter the input.

        sigmas: the standard deviations of each feature in the filter.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, features, sigmas=None):

        scaled_features = features
        if sigmas is not None:
            for i in range(features.size(1)):
                scaled_features[:, i, ...] /= sigmas[i]

        ctx.save_for_backward(scaled_features)
        output_data = _C.phl_filter(input, scaled_features)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("PHLFilter does not currently support Backpropagation")
        # scaled_features, = ctx.saved_variables
        # grad_input = _C.phl_filter(grad_output, scaled_features)
        # return grad_input


class TrainableBilateralFilterFunction(torch.autograd.Function):
    """
    torch.autograd.Function for the TrainableBilateralFilter layer.

    See:
        F. Wagner, et al., Ultralow-parameter denoising: Trainable bilateral filter layers in
        computed tomography, Medical Physics (2022), https://doi.org/10.1002/mp.15718

    Args:
        input: input tensor to be filtered.

        sigma x: trainable standard deviation of the spatial filter kernel in x direction.

        sigma y: trainable standard deviation of the spatial filter kernel in y direction.

        sigma z: trainable standard deviation of the spatial filter kernel in z direction.

        color sigma: trainable standard deviation of the intensity range kernel. This filter
            parameter determines the degree of edge preservation.

    Returns:
        output (torch.Tensor): filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, color_sigma):
        output_tensor, output_weights_tensor, do_dx_ki, do_dsig_r, do_dsig_x, do_dsig_y, do_dsig_z = _C.tbf_forward(
            input_img, sigma_x, sigma_y, sigma_z, color_sigma
        )

        ctx.save_for_backward(
            input_img,
            sigma_x,
            sigma_y,
            sigma_z,
            color_sigma,
            output_tensor,
            output_weights_tensor,
            do_dx_ki,
            do_dsig_r,
            do_dsig_x,
            do_dsig_y,
            do_dsig_z,
        )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        output_tensor = ctx.saved_tensors[5]  # filtered image
        output_weights_tensor = ctx.saved_tensors[6]  # weights
        do_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        do_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        do_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        do_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        do_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * do_dsig_r)
        grad_sig_x = torch.sum(grad_output * do_dsig_x)
        grad_sig_y = torch.sum(grad_output * do_dsig_y)
        grad_sig_z = torch.sum(grad_output * do_dsig_z)

        grad_output_tensor = _C.tbf_backward(
            grad_output,
            input_img,
            output_tensor,
            output_weights_tensor,
            do_dx_ki,
            sigma_x,
            sigma_y,
            sigma_z,
            color_sigma,
        )

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class TrainableBilateralFilter(torch.nn.Module):
    """
    Implementation of a trainable bilateral filter layer as proposed in the corresponding publication.
    All filter parameters can be trained data-driven. The spatial filter kernels x, y, and z determine
    image smoothing whereas the color parameter specifies the amount of edge preservation.
    Can run on 1D, 2D, or 3D tensors (on top of Batch and Channel dimensions).

    See:
        F. Wagner, et al., Ultralow-parameter denoising: Trainable bilateral filter layers in
        computed tomography, Medical Physics (2022), https://doi.org/10.1002/mp.15718

    Args:
        input: input tensor to be filtered.

        sigma x: trainable standard deviation of the spatial filter kernel in x direction.

        sigma y: trainable standard deviation of the spatial filter kernel in y direction.

        sigma z: trainable standard deviation of the spatial filter kernel in z direction.

        sigma color: trainable standard deviation of the intensity range kernel. This filter
            parameter determines the degree of edge preservation.

    Returns:
        output (torch.Tensor): filtered tensor.
    """

    def __init__(self, sigma_x, sigma_y, sigma_z, sigma_color):
        super().__init__()

        # Register sigmas as trainable parameters.
        self.sigma_x = torch.nn.Parameter(torch.tensor(sigma_x))
        self.sigma_y = torch.nn.Parameter(torch.tensor(sigma_y))
        self.sigma_z = torch.nn.Parameter(torch.tensor(sigma_z))
        self.sigma_color = torch.nn.Parameter(torch.tensor(sigma_color))

    def forward(self, input_tensor):
        assert input_tensor.shape[1] == 1, (
            "Currently channel dimensions >1 are not supported. "
            "Please use multiple parallel filter layers if you want "
            "to filter multiple channels."
        )

        len_input = len(input_tensor.shape)

        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)

        prediction = TrainableBilateralFilterFunction.apply(
            input_tensor, self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_color
        )

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            prediction = prediction.squeeze(4).squeeze(3)
        elif len_input == 4:
            prediction = prediction.squeeze(4)

        return prediction
