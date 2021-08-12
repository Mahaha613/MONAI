# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import Compose, Flip, RandFlip, RandFlipD, Randomizable, ToTensor, ToTensorD
from monai.transforms.nvtx import (
    Mark,
    MarkD,
    RandMark,
    RandMarkD,
    RandRangePop,
    RandRangePopD,
    RandRangePush,
    RandRangePushD,
    RangePop,
    RangePopD,
    RangePush,
    RangePushD,
)
from monai.utils import optional_import

_, has_nvtx = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")


TEST_CASE_ARRAY_0 = [
    np.random.randn(3, 3),
]
TEST_CASE_ARRAY_1 = [
    np.random.randn(3, 10, 10),
]
TEST_CASE_DICT_0 = [
    {"image": np.random.randn(3, 3)},
]
TEST_CASE_DICT_1 = [
    {"image": np.random.randn(3, 10, 10)},
]


class TestNVTXTransforms(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_ARRAY_0,
            TEST_CASE_ARRAY_1,
            TEST_CASE_DICT_0,
            TEST_CASE_DICT_1,
        ]
    )
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX!")
    def test_nvtx_transfroms_alone(self, input):
        transforms = Compose(
            [
                Mark("Mark: Transform Starts!"),
                RangePush("Range: RandFlipD"),
                RangePop(),
                RandRangePush("Range: ToTensorD"),
                RandRangePop(),
                RandMark("Mark: Transform Ends!"),
            ]
        )
        output = transforms(input)
        self.assertEqual(id(input), id(output))

        # Check if chain of randomizable/non-randomizable transforms is not broken
        for tran in transforms.transforms:
            if isinstance(tran, Randomizable):
                self.assertIsInstance(tran, RangePush)
                break

    @parameterized.expand([TEST_CASE_ARRAY_0, TEST_CASE_ARRAY_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX!")
    def test_nvtx_transfroms_array(self, input):
        transforms = Compose(
            [
                RandMark("Mark: Transform Starts!"),
                RandRangePush("Range: RandFlip"),
                RandFlip(prob=0.0),
                RandRangePop(),
                RangePush("Range: ToTensor"),
                ToTensor(),
                RangePop(),
                Mark("Mark: Transform Ends!"),
            ]
        )
        output = transforms(input)
        self.assertIsInstance(output, torch.Tensor)
        np.testing.assert_array_equal(input, output)

        transforms = Compose(
            [
                RandMark("Mark: Transform Starts!"),
                RandRangePush("Range: RandFlip"),
                RandFlip(prob=1.0),
                RandRangePop(),
                RangePush("Range: ToTensor"),
                ToTensor(),
                RangePop(),
                Mark("Mark: Transform Ends!"),
            ]
        )
        output = transforms(input)
        self.assertIsInstance(output, torch.Tensor)
        np.testing.assert_array_equal(input, Flip()(output.numpy()))

    @parameterized.expand([TEST_CASE_DICT_0, TEST_CASE_DICT_1])
    @unittest.skipUnless(has_nvtx, "CUDA is required for NVTX!")
    def test_nvtx_transfromsd(self, input):
        transforms = Compose(
            [
                RandMarkD("Mark: Transform Starts!"),
                RandRangePushD("Range: RandFlipD"),
                RandFlipD(keys="image", prob=0.0),
                RandRangePopD(),
                RangePushD("Range: ToTensorD"),
                ToTensorD(keys=("image")),
                RangePopD(),
                MarkD("Mark: Transform Ends!"),
            ]
        )
        output = transforms(input)
        self.assertIsInstance(output["image"], torch.Tensor)
        np.testing.assert_array_equal(input["image"], output["image"])

        transforms = Compose(
            [
                RandMarkD("Mark: Transform Starts!"),
                RandRangePushD("Range: RandFlipD"),
                RandFlipD(keys="image", prob=1.0),
                RandRangePopD(),
                RangePushD("Range: ToTensorD"),
                ToTensorD(keys=("image")),
                RangePopD(),
                MarkD("Mark: Transform Ends!"),
            ]
        )
        output = transforms(input)
        self.assertIsInstance(output["image"], torch.Tensor)
        np.testing.assert_array_equal(input["image"], Flip()(output["image"].numpy()))


if __name__ == "__main__":
    unittest.main()
