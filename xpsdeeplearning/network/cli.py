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
import os
import json
import datetime
import pytz
import click
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.losses as tf_losses
import tensorflow.keras.metrics as tf_metrics

from xpsdeeplearning.network.classifier import Classifier
from xpsdeeplearning.network import models


def init_clf(exp_params):
    """Init a blank classifier."""
    return Classifier(
        time=datetime.datetime.now()
        .astimezone(pytz.timezone("Europe/Berlin"))
        .strftime("%Y%m%d_%Hh%Mm"),
        exp_name=exp_params["exp_name"],
        task=exp_params["task"],
        intensity_only=exp_params["intensity_only"],
    )


def init_clf_with_data(exp_params: dict):
    """Init a classifier and load some data."""
    clf = init_clf(exp_params)

    input_filepath = exp_params["input_filepath"]
    no_of_examples = exp_params["no_of_examples"]
    train_test_split = exp_params["train_test_split"]
    train_val_split = exp_params["train_val_split"]

    _ = clf.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )
    _ = clf.datahandler.check_class_distribution(clf.task)

    return clf


def select_model_class(task: str = "regression"):
    """Select the correct pre-defined model for a given task."""
    if task == "regression":
        return models.RegressionCNN
    if task == "classification":
        return models.ClassificationCNN
    if task == "multi_class_detection":
        return models.ClassificationCNN
    return models.RegressionCNN


def select_loss_and_metrics(task: str = "regression"):
    """Select loss and metric for a given task."""
    if task == "regression":
        loss = tf_losses.MeanAbsoluteError()
        metrics = [tf_metrics.MeanSquaredError(name="mse")]
    elif task == "classification":
        loss = tf_losses.CategoricalCrossentropy()
        metrics = [tf_metrics.CategoricalCrossentropy(name="accuracy")]
    elif task == "multi_class_detection":
        loss = tf_losses.BinaryCrossentropy()
        metrics = [tf_metrics.BinaryAccuracy(name="accuracy")]

    return loss, metrics


@click.command()
@click.option(
    "--param-file",
    default=None,
    required=True,
    help="The path to the training parameter file.",
)
def train_cli(param_file: str):
    """Train a CNN on new data."""
    with open(param_file, "r") as json_file:
        training_params = json.load(json_file)

    clf = init_clf_with_data(training_params)

    model_class = select_model_class(clf.task)
    clf.model = model_class(clf.datahandler.input_shape, clf.datahandler.num_classes)

    loss, metrics = select_loss_and_metrics(clf.task)
    learning_rate = training_params["learning_rate"]

    clf.model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics,
    )
    clf.summary()

    _ = clf.train(
        checkpoint=True,
        early_stopping=False,
        tb_log=True,
        csv_log=True,
        hyperparam_log=True,
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        verbose=1,
    )
    if clf.task == "regression":
        test_loss = clf.evaluate()
        print("Test loss: " + str(np.round(test_loss, decimals=8)))

    else:
        score = clf.evaluate()
        test_loss, test_accuracy = score[0], score[1]
        print("Test loss: " + str(np.round(test_loss, decimals=8)))
        print("Test accuracy: " + str(np.round(test_accuracy, decimals=3)))

    _ = clf.predict()
    if clf.task != "regression":
        _ = clf.predict_classes()

    clf.pickle_results()


@click.command()
@click.option(
    "--param-file",
    default=None,
    required=True,
    help="The path to the parameter file for this experiment.",
)
@click.option(
    "--clf-path",
    default=None,
    required=True,
    help="The path to the existing classifier.",
)
def predict_cli(param_file: str, clf_path: str):
    """Predict using an existing classifier."""
    with open(param_file, "r") as json_file:
        test_params = json.load(json_file)
    clf_param_file = os.path.join(clf_path, "training_params.json")
    with open(clf_param_file, "r") as clf_json_file:
        clf_train_params = json.load(clf_json_file)

    clf = init_clf_with_data(test_params)
    clf.load_model(model_path=clf_path)
    clf.summary()

    clf.logging.hyperparams["batch_size"] = clf_train_params["train_params"][
        "batch_size"
    ]

    if clf.task == "regression":
        test_loss = clf.evaluate()
        print("Test loss: " + str(np.round(test_loss, decimals=8)))

    else:
        score = clf.evaluate()
        test_loss, test_accuracy = score[0], score[1]
        print("Test loss: " + str(np.round(test_loss, decimals=8)))
        print("Test accuracy: " + str(np.round(test_accuracy, decimals=3)))

    _ = clf.predict()
    if clf.task != "regression":
        _ = clf.predict_classes()
    clf.pickle_results()
