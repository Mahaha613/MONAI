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

import unittest

import numpy as np
from parameterized import parameterized
from monai.transforms import NormalizeIntensityd
from tests.utils import NumpyImageTestCase2D

TEST_CASE_1 = [
    {"keys": ["img"], "nonzero": True},
    {"img": np.array([0.0, 3.0, 0.0, 4.0])},
    np.array([0.0, -1.0, 0.0, 1.0]),
]

TEST_CASE_2 = [
    {
        "keys": ["img"],
        "subtrahend": np.array([3.5, 3.5, 3.5, 3.5]),
        "divisor": np.array([0.5, 0.5, 0.5, 0.5]),
        "nonzero": True,
    },
    {"img": np.array([0.0, 3.0, 0.0, 4.0])},
    np.array([0.0, -1.0, 0.0, 1.0]),
]

TEST_CASE_3 = [
    {"keys": ["img"], "nonzero": True},
    {"img": np.array([0.0, 0.0, 0.0, 0.0])},
    np.array([0.0, 0.0, 0.0, 0.0]),
]


class TestNormalizeIntensityd(NumpyImageTestCase2D):
    def test_image_normalize_intensityd(self):
        key = "img"
        normalizer = NormalizeIntensityd(keys=[key])
        normalized = normalizer({key: self.imt})
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        np.testing.assert_allclose(normalized[key], expected, rtol=1e-6)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_nonzero(self, input_param, input_data, expected_data):
        normalizer = NormalizeIntensityd(**input_param)
        np.testing.assert_allclose(expected_data, normalizer(input_data)["img"])

    def test_channel_wise(self):
        key = "img"
        normalizer = NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True)
        input_data = {key: np.array([[0.0, 3.0, 0.0, 4.0], [0.0, 4.0, 0.0, 5.0]])}
        expected = np.array([[0.0, -1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0]])
        np.testing.assert_allclose(expected, normalizer(input_data)[key])


if __name__ == "__main__":
    unittest.main()
