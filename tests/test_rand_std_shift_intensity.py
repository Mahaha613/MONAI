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

from monai.transforms import RandStdShiftIntensity
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D


class TestRandStdShiftIntensity(NumpyImageTestCase2D):
    def test_value(self):
        for p in TEST_NDARRAYS:
            np.random.seed(0)
            # simulate the randomize() of transform
            np.random.random()
            factor = np.random.uniform(low=-1.0, high=1.0)
            offset = factor * np.std(self.imt)
            expected = p(self.imt + offset)
            shifter = RandStdShiftIntensity(factors=1.0, prob=1.0)
            shifter.set_random_state(seed=0)
            result = shifter(p(self.imt))
            torch.testing.assert_allclose(result, expected, atol=0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
