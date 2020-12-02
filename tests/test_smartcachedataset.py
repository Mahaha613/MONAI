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
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import SmartCacheDataset
from monai.transforms import Compose, LoadImaged

TEST_CASE_1 = [0.1, 0, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_2 = [0.1, 4, Compose([LoadImaged(keys=["image", "label", "extra"])])]

TEST_CASE_3 = [0.1, 4, None]

TEST_CASE_4 = [0.5, 2, Compose([LoadImaged(keys=["image", "label", "extra"])])]


class TestSmartCacheDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_shape(self, replace_rate, num_replace_workers, transform):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[8, 8, 8]), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
            test_data = [
                {
                    "image": os.path.join(tempdir, "test_image1.nii.gz"),
                    "label": os.path.join(tempdir, "test_label1.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra1.nii.gz"),
                }
            ] * 20
            dataset = SmartCacheDataset(
                data=test_data,
                transform=transform,
                replace_rate=replace_rate,
                cache_num=16,
                num_init_workers=4,
                num_replace_workers=num_replace_workers,
            )

            self.assertEqual(len(dataset._cache), dataset.cache_num)
            for i in range(dataset.cache_num):
                self.assertIsNotNone(dataset._cache[i])

            for _ in range(2):
                dataset.start()
                for _ in range(3):
                    dataset.update_cache()
                    self.assertIsNotNone(dataset._cache[15])
                    if isinstance(dataset._cache[15]["image"], np.ndarray):
                        np.testing.assert_allclose(dataset._cache[15]["image"], dataset._cache[15]["label"])
                    else:
                        self.assertIsInstance(dataset._cache[15]["image"], str)
                dataset.shutdown()


if __name__ == "__main__":
    unittest.main()
