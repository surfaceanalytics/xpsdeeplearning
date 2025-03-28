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
Tests for classification and prediction.
"""

import os
import re
import sys
import csv
import json

import pickle
import shutil

import numpy as np
import pytest
from click.testing import CliRunner
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from xpsdeeplearning.network.classifier import Classifier
from xpsdeeplearning.network.cli import train_cli, predict_cli
from xpsdeeplearning.network.models import RegressionCNN
from xpsdeeplearning.network.prepare_upload import Uploader


def init_clf():
    """Init a blank classifier."""
    return Classifier(
        time="20240101_00h00m",
        exp_name="test",
        task="regression",
        intensity_only=True,
        labels=[
            "Ni metal",
            "NiO",
        ],
    )


def init_clf_with_data():
    """Init a classifier and load some data."""
    clf = init_clf()

    input_filepath = "tests/data/20240206_Ni_linear_combination_small_gas_phase.h5"
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 20

    _ = clf.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )

    return clf


def del_clf_dirs(clf):
    """Remove classifier dirs (to be used after test)."""
    shutil.rmtree(clf.logging.model_dir)
    shutil.rmtree(clf.logging.log_dir)
    shutil.rmtree(clf.logging.fig_dir)


def load_reference_model(clf):
    """Load a trained reference TF model."""
    ref_model_dir = "tests/data/clf/test_clf/model"
    clf.load_model(model_path=ref_model_dir)


@pytest.mark.parametrize(
    "init_func",
    [
        pytest.param(init_clf, id="without_data"),
        pytest.param(init_clf_with_data, id="with_data"),
    ],
)
def test_init_clf(init_func):
    """Test for classifier initialization."""
    clf = init_func()
    ref_log_dir = "tests/data/clf/test_clf/logs"

    hyperparams = []

    for log_dir in [clf.logging.log_dir, ref_log_dir]:
        hyperparam_file_name = os.path.join(log_dir, "hyperparameters.json")
        with open(hyperparam_file_name, "r") as json_file:
            hyperparams += [json.load(json_file)]

    for key, value in hyperparams[0].items():
        if key not in ["epochs_trained"]:
            assert value == hyperparams[1][key]

    del_clf_dirs(clf)
    sys.stdout.write("Test on classifier init okay.\n")


def test_load_model():
    """Test for classifier model loading."""
    clf = init_clf_with_data()
    load_reference_model(clf)
    clf.summary()

    ref_model_summary_file = "tests/data/clf/test_clf/model/model_summary.txt"

    with open(ref_model_summary_file, "r") as file:
        ref_model_summary = file.read()

    assert clf.model_summary == ref_model_summary
    del_clf_dirs(clf)
    sys.stdout.write("Test for classifier model loading okay.\n")


def test_train():
    """Test for classifier model training."""

    def assert_dict_similarity(dict1, dict2, tol=0.01):
        assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"

        for key in dict1:
            arr1 = np.array(dict1[key])
            arr2 = np.array(dict2[key])

            assert arr1.shape == arr2.shape, f"Shape mismatch for key '{key}'"
            assert np.allclose(arr1, arr2, atol=tol), f"Values differ for key '{key}'"

    clf = init_clf_with_data()

    clf.model = RegressionCNN(clf.datahandler.input_shape, clf.datahandler.num_classes)

    clf.model.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=1e-05),
        metrics=[MeanSquaredError(name="mse")],
    )

    hist = clf.train(
        epochs=3,
        batch_size=32,
        checkpoint=False,
        early_stopping=False,
        tb_log=False,
        csv_log=True,
        verbose=1,
    )

    ref_log_dir = "tests/data/clf/test_clf/logs"

    ref_csv_file = os.path.join(ref_log_dir, "log.csv")
    ref_hyperparams_file = os.path.join(ref_log_dir, "hyperparameters.json")

    history = {}
    with open(ref_csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, item in row.items():
                if key not in history.keys():
                    history[key] = []
                history[key].append(float(item))

    assert_dict_similarity(hist, history)

    del_clf_dirs(clf)
    sys.stdout.write("Test on classifier training okay.\n")


def test_evaluate():
    """Test for classifier evaluation."""
    clf = init_clf_with_data()
    load_reference_model(clf)

    test_loss, test_accuracy = clf.evaluate()

    assert np.around(test_loss, 3) == 0.394
    assert np.around(test_accuracy, 7) == 0.1889244

    del_clf_dirs(clf)
    sys.stdout.write("Test on classifier evaluation okay.\n")


def test_predict():
    """Test for classifier prediction."""
    clf = init_clf_with_data()
    load_reference_model(clf)
    clf.predict()

    ref_pred_file = "tests/data/clf/test_clf/logs/results.pkl"

    with open(ref_pred_file, "rb") as pickle_file:
        ref_pred = pickle.load(pickle_file)

    np.testing.assert_array_almost_equal(
        clf.datahandler.pred_train, ref_pred["pred_train"], decimal=8
    )
    np.testing.assert_array_almost_equal(
        clf.datahandler.pred_test, ref_pred["pred_test"], decimal=8
    )

    del_clf_dirs(clf)
    sys.stdout.write("Test on classifier predicition okay.\n")


def test_upload():
    """Test for classifier evaluation."""
    clf_root_dir = "tests/data/clf/test_clf/"

    hyperparam_file_name = "tests/data/clf/test_clf/logs/hyperparameters.json"
    with open(hyperparam_file_name, "r") as json_file:
        hyperparams = json.load(json_file)

    dataset_path = hyperparams["input_filepath"].rsplit(".", 1)[0] + "_metadata.json"

    uploader = Uploader(clf_root_dir, dataset_path)
    uploader.prepare_upload_params()

    ref_upload_file = "tests/data/clf/test_clf/ref_upload_params.json"

    with open(ref_upload_file, "r") as ref_json_file:
        ref_upload_params = json.load(ref_json_file)

    assert uploader.upload_params == ref_upload_params
    sys.stdout.write("Test on upload okay.\n")


@pytest.mark.parametrize(
    "cli_inputs, ref_file",
    [
        pytest.param(
            [
                train_cli,
                "--param-file",
                "tests/data/clf/train_params.json",
            ],
            "tests/data/clf/ref_output_train.txt",
            id="train",
        ),
        pytest.param(
            [
                predict_cli,
                "--param-file",
                "tests/data/clf/test_params.json",
                "--model-path",
                "tests/data/clf/test_clf/model",
            ],
            "tests/data/clf/ref_output_test.txt",
            id="predict",
        ),
    ],
)
def test_cli(cli_inputs, ref_file):
    """Test CLI functions for training and prediction."""

    def assert_loss_within_tolerance(
        pattern: re.Pattern, line: str, ref_line: str, tolerance: float = 0.05
    ):
        """Compare loss and mse values between actual and reference lines within a given tolerance."""
        match_output = pattern.search(line)
        match_expected = pattern.search(ref_line)

        if match_output and match_expected:
            # Extract floating point numbers
            values_output = [float(x) for x in match_output.groups()]
            values_expected = [float(x) for x in match_expected.groups()]

            # Ensure both lists have the same number of values
            assert len(values_output) == len(values_expected), (
                "Mismatched loss/mse value count!"
            )

            # Check if each value is within the allowed tolerance
            for out, exp in zip(values_output, values_expected):
                assert abs(out - exp) / exp <= tolerance, (
                    f"Value {out} not within {tolerance * 100}% of {exp}"
                )
        else:
            # Fallback: If no loss/mse match, compare the lines directly
            assert line.strip() == ref_line.strip(), (
                f"Line mismatch:\nExpected: {ref_line}\nActual: {line}"
            )

    runner = CliRunner()
    result = runner.invoke(cli_inputs[0], cli_inputs[1:])

    assert result.exit_code == 0

    shutil.rmtree("runs/20250101_Ni_small_gas_phase")

    output = result.stdout.split("\n")

    with open(ref_file, "r") as file:
        expected_messages = [line.strip() for line in file.readlines()]

    # Regular expressions for matching losses and MSEs
    loss_mse_pattern = re.compile(r"loss: ([\d.]+) - mse: ([\d.]+)")
    loss_mse_val_pattern = re.compile(
        r"loss: ([\d.]+) - mse: ([\d.]+) - val_loss: ([\d.]+) - val_mse: ([\d.]+)"
    )
    test_loss_pattern = re.compile(r"Test loss: \[([\d.]+) ([\d.]+)\]")

    tolerance = 0.05  # 5% tolerance

    for line, ref_line in zip(output, expected_messages):
        if loss_mse_pattern.search(line):
            try:
                assert_loss_within_tolerance(
                    loss_mse_pattern, line, ref_line, tolerance
                )
            except AssertionError:
                assert_loss_within_tolerance(
                    loss_mse_val_pattern, line, ref_line, tolerance
                )
        elif test_loss_pattern.search(line):
            assert_loss_within_tolerance(test_loss_pattern, line, ref_line, tolerance)
        else:
            if "ms/step" in line:
                continue
            assert line.strip() == ref_line.strip()

    sys.stdout.write(f"Test on {str(cli_inputs[0])} okay.\n")
