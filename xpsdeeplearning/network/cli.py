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
#
"""
CLI tools for classification and prediction.
"""
import json
import datetime
import pytz
import click

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError

from xpsdeeplearning.network.classifier import Classifier
from xpsdeeplearning.network.models import RegressionCNN


def init_clf(exp_params):
    """Init a blank classifier."""
    return Classifier(
        time=datetime.datetime.now()
        .astimezone(pytz.timezone("Europe/Berlin"))
        .strftime("%Y%m%d_%Hh%Mm"),
        exp_name=exp_params["exp_name"],
        task=exp_params["task"],
        intensity_only=exp_params["intensity_only"],
        labels=exp_params["labels"],
    )


def init_clf_with_data(input_filepath, exp_params):
    """Init a classifier and load some data."""
    clf = init_clf(exp_params)

    input_filepath = input_filepath
    train_test_split = exp_params["train_test_split"]
    train_val_split = exp_params["train_test_split"]
    no_of_examples = exp_params["no_of_examples"]

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        aug_values_train,
        aug_values_val,
        aug_values_test,
    ) = clf.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )

    return clf


@click.command()
@click.option(
    "--hdf5-file",
    default=None,
    required=True,
    type=click.File("r"),
    help="The path to the HDF5 file with the training data.",
)
@click.option(
    "--exp-param-file",
    default=None,
    required=True,
    type=click.File("r"),
    help="The path to the training parameter file.",
)
@click.option(
    "--output-folder",
    default=None,
    help="The path to the output folder of the training results.",
)
def train_cli(hdf5_file: str, exp_param_file: str, output_folder: str):
    with open(exp_param_file, "r") as json_file:
        training_params = json.load(json_file)

    ## open training params
    clf = init_clf_with_data(hdf5_file, training_params)
    clf.model = RegressionCNN(clf.datahandler.input_shape, clf.datahandler.num_classes)

    clf.model.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=1e-05),
        metrics=[MeanSquaredError(name="mse")],
    )

    hist = clf.train(
        epochs=1,
        batch_size=32,
        checkpoint=False,
        early_stopping=False,
        tb_log=False,
        csv_log=True,
        verbose=1,
    )
    clf.save_results()


@click.option(
    "--hdf5-file",
    default=None,
    required=True,
    type=click.File("r"),
    help="The path to the HDF5 file with the test data.",
)
@click.option(
    "--exp-param-file",
    default=None,
    required=True,
    type=click.File("r"),
    help="The path to the parameter file for this experiment.",
)
@click.option(
    "--clf_path",
    default=None,
    required=True,
    type=click.File("r"),
    help="The path to the existing classifier.",
)
def predict_cli(hdf5_file: str, exp_param_file: str, clf_path: str):
    """Predict using an existing classifier."""
    with open(exp_param_file, "r") as json_file:
        test_params = json.load(json_file)

    clf = init_clf_with_data(hdf5_file, test_params)
    clf.load_model(model_path=clf_path)

    test_loss = clf.evaluate()
    pred_train, pred_test = clf.predict()
    clf.save_results()
