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
Prepare classsifier for website upload.
"""

import os
import json
import shutil


class Uploader:
    """Class for preparing the upload of a classifier."""

    def __init__(self, model_path, dataset_metadata_path):
        """
        Initialize the two relevant paths.

        Parameters
        ----------
        model_path : str
            Path of the classifier.
        dataset_metadata_path : str
            Path of the JSON metadata file for the data set.

        Returns
        -------
        None.

        """
        self.model_path = model_path
        self.dataset_metadata_path = dataset_metadata_path

        self.upload_params = {}

    def _get_train_params(self):
        """
        Get all relevant parameters from the training log.

        Returns
        -------
        train_params : dict
            Dictionary with all relevant training log data.

        """
        log_path = os.path.join(
            *[
                self.model_path,
                "logs",
                "hyperparameters.json",
            ]
        )

        with open(log_path, "r") as param_file:
            train_params = json.load(param_file)

        # Remove unneeded information
        for key in [
            "input_filepath",
            "model_summary",
            "No. of training samples",
            "No. of validation samples",
            "No. of test samples",
            "Shape of each sample",
            "train_test_split",
            "train_val_split",
            "num_of_classes",
        ]:
            del train_params[key]

        return train_params

    def _get_data_params(self):
        """
        Get all relevant parameters from the dataset metadata.

        Returns
        -------
        data_params : dict
            Dictionary with all relevant metadata from the dataset
            creation.

        """
        with open(self.dataset_metadata_path, "r") as param_file:
            data_params = json.load(param_file)

        # Remove unneeded information
        for key in [
            "single",
            "variable_no_of_inputs",
            "no_of_simulations",
            "labels",
            "output_datafolder",
        ]:
            del data_params[key]

        return data_params

    def prepare_upload_params(
        self,
    ):
        """
        Create nested dictionary for the classifier website upload.

        Returns
        -------
        None.

        """
        train_params = self._get_train_params()
        data_params = self._get_data_params()

        name = train_params["time"] + "_" + train_params["exp_name"]
        labels = train_params["labels"]

        for key in ["time", "exp_name", "labels"]:
            del train_params[key]

        self.upload_params = {
            "name": name,
            "labels": labels,
            "dataset_params": data_params,
            "train_params": train_params,
        }

    def save_upload_params(self):
        """
        Save the nested dictionary to a new json file.

        Returns
        -------
        None.

        """
        upload_param_file = os.path.join(self.model_path, "upload_params.json")

        with open(upload_param_file, "w", encoding="utf-8") as json_file:
            json.dump(
                self.upload_params,
                json_file,
                ensure_ascii=False,
                indent=4,
            )

        print("JSON file was prepared for upload!")

    def zip_all(self):
        """
        Create a zip file of the complete experiment.

        Returns
        -------
        None.

        """
        shutil.make_archive(
            self.model_path,
            "zip",
            self.model_path,
        )
