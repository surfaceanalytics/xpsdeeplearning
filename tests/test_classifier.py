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
import sys
import csv
import json
import pytest
import numpy as np
from click.testing import CliRunner

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError

from xpsdeeplearning.network.classifier import Classifier
from xpsdeeplearning.network.models import RegressionCNN
from xpsdeeplearning.network.prepare_upload import Uploader
from xpsdeeplearning.network.cli import train_cli, predict_cli


def init_clf():
    """Init a blank classifier."""
    return Classifier(
        time="20240101_00h00m",
        exp_name="test",
        task="regression",
        intensity_only=True,
        labels=["Ni metal", "NiO",],
        )

def init_clf_with_data():
    """Init a classifier and load some data."""
    clf = init_clf()

    input_filepath = "tests/data/20240202_Ni_linear_combination_small_gas_phase.h5"
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 200

    X_train, X_val, X_test, y_train, y_val, y_test, aug_values_train, aug_values_val, aug_values_test =\
        clf.load_data_preprocess(
            input_filepath = input_filepath,
            no_of_examples = no_of_examples,
            train_test_split = train_test_split,
            train_val_split = train_val_split
        )

    return clf

def del_clf_dirs(clf):
    """Remove classifier dirs (to be used after test)."""
    os.remove(clf.logging.model_dir)
    os.remove(clf.logging.log_dir)
    os.remove(clf.logging.fig_dir)

def load_reference_model(clf):
    """Load a trained reference TF model."""
    ref_model_dir = "tests/data/test_clf/model" #########
    return clf.load_model(model_path=ref_model_dir)


def test_init_clf():
    """Test for classifier initialization."""
    clf = init_clf()
    ref_log_dir = "tests/data/test_clf/logs"

    hyperparams = []

    for log_dir in [ref_log_dir, clf.logging.log_dir]:
        hyperparam_file_name = os.path.join(
            log_dir, "hyperparameters.json"
        )
    with open(hyperparam_file_name, "r") as json_file:
        hyperparams += [json.load(json_file)]

    assert hyperparams[0] == hyperparams[1]
    del_clf_dirs(clf)
    sys.stdout.write("Test on classifier init okay.\n")

def test_load_model():
    """Test for classifier model loading."""
    clf = init_clf()
    clf.model = load_reference_model(clf)

    ref_model_summary_file = "tests/data/test_clf/model/model_summary.txt"        #########
    with open(ref_model_summary_file, "r") as file:
        ref_model_summary = file.read()

    assert clf.model_summary == ref_model_summary
    del_clf_dirs()
    sys.stdout.write("Test for classifier model loading okay.\n")

def test_train():
    """Test for classifier model training."""
    clf = init_clf_with_data()

    clf.model = RegressionCNN(
        clf.datahandler.input_shape,
        clf.datahandler.num_classes)

    clf.model.compile(loss=MeanAbsoluteError(),
                  optimizer=Adam(learning_rate=1e-05),
                  metrics=[MeanSquaredError(name="mse")])

    hist = clf.train(
        epochs=1,
        batch_size=32,
        checkpoint=False,
        early_stopping=False,
        tb_log=False,
        csv_log=True,
        verbose = 1)

    ref_log_dir = "tests/data/test_clf/logs"

    ref_csv_file = os.path.join(ref_log_dir, "log.csv")

    history = {}
    with open(ref_csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, item in row.items():
                if key not in history.keys():
                    history[key] = []
                history[key].append(float(item))

    assert hist == history
    del_clf_dirs()
    sys.stdout.write("Test on classifier training okay.\n")

def test_evaluate():
    """Test for classifier evaluation."""
    clf = init_clf_with_data()
    clf.model = load_reference_model(clf)

    test_loss = clf.evaluate()

    assert test_loss == 0.0
    del_clf_dirs()
    sys.stdout.write("Test on classifier evaluation okay.\n")

def test_predict():
    """Test for classifier predicition."""
    clf = init_clf_with_data()
    clf.model = load_reference_model(clf)

    pred_train, pred_test = clf.predict()

    ref_pred_file = "tests/data/ref_pred_test.npz" ######

    #np.savez_compressed('filename.npz', array1=array1, array2=array2)
    ref_pred_test = np.load(ref_pred_file)

    assert clf.pred_test == ref_pred_test
    del_clf_dirs()
    sys.stdout.write("Test on classifier predicition okay.\n")

def test_upload():
    """Test for classifier evaluation."""
    clf_root_dir = "tests/data/test_clf/"


    hyperparam_file_name = "tests/data/test_clf/logs/hyperparameters.json"
    with open(hyperparam_file_name, "r") as json_file:
        hyperparams = json.load(json_file)

    dataset_path = hyperparams["input_filepath"].rsplit(".",1)[0] + "_metadata.json"

    uploader = Uploader(clf_root_dir, dataset_path)
    uploader.prepare_upload_params()

    ref_upload_file = "tests/data/test_clf/ref_upload_params"
    with open(ref_upload_file, "r") as ref_json_file:
        ref_upload_params = json.load(ref_json_file)


    assert uploader.upload_params == ref_upload_params
    sys.stdout.write("Test on upload okay.\n")


@pytest.mark.parametrize(
    "cli_inputs",
    [
        pytest.param(
            [
                "--hdf5_file",
                "tests/data/20240202_Ni_linear_combination_small_gas_phase.h5"
                "--exp-param-file",
                "tests/data/exp_params_train.json",
                "--output-folder",
                "tests/data/clf_train",
            ],
        ),
    ],
)
def test_train_cli(cli_inputs):
    runner = CliRunner()
    result = runner.invoke(train_cli, cli_inputs)

    assert result.exit_code == 0
    #os.remove(hdf5_file)
    sys.stdout.write("Test on train_cli okay.\n")

@pytest.mark.parametrize(
    "cli_inputs",
    [
        pytest.param(
            [
                "--hdf5_file",
                "tests/data/20240202_Ni_linear_combination_small_gas_phase.h5"
                "--exp-param-file",
                "tests/data/exp_params_test.json",
                "--output-folder",
                "tests/data/clf_test",
            ],
        ),
    ],
)
def test_predict_cli(cli_inputs):
    runner = CliRunner()
    result = runner.invoke(predict_cli, cli_inputs)

    assert result.exit_code == 0
    os.remove("tests/data/clf_test")
    sys.stdout.write("Test on predict_cli okay.\n")
