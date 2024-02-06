#
# Copyright the xpsdeeplearning authors.
#
# This file is part of xpsdeeplearning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for data loader.
"""
import sys

import numpy as np

from xpsdeeplearning.network.data_handling import DataHandler


def test_hdf5_load():
    np.random.seed(1)

    input_filepath = "tests/data/20240206_Ni_linear_combination_small_gas_phase.h5"

    datahandler = DataHandler(intensity_only=False)
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 20

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        sim_values_train,
        sim_values_val,
        sim_values_test,
    ) = datahandler.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
        select_random_subset=False,
        shuffle=False,
    )
    print("Input shape: " + str(datahandler.input_shape))
    print("Labels: " + str(datahandler.labels))
    print("No. of classes: " + str(datahandler.num_classes))

    assert X_train.shape == (12, 501, 2)
    assert X_val.shape == (4, 501, 2)
    assert X_test.shape == (4, 501, 2)
    assert y_train.shape == (12, 2)
    assert y_val.shape == (4, 2)
    assert y_test.shape == (4, 2)
    sys.stdout.write("HDF5 loading test okay.\n")