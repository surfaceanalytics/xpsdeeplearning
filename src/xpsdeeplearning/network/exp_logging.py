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
Logging for training and testing a Keras model.
"""
import os
import json
import csv
import shutil

import tensorflow as tf
from tensorflow.keras import callbacks


class ExperimentLogging:
    """Class for logigng during an experiment in tensorflow."""

    def __init__(self, dir_name):
        """
        Initialize folder structure and keras Callbacks.

        Parameters
        ----------
        dir_name : str
            Name of the folders to create.

        Returns
        -------
        None.

        """
        self.root_dir = os.path.join(*[os.getcwd(), "runs", dir_name])

        self.model_dir = os.path.join(self.root_dir, "model")
        self.log_dir = os.path.join(self.root_dir, "logs")
        self.fig_dir = os.path.join(self.root_dir, "figures")

        ### Callbacks ###
        self.active_cbs = []

        # Save the model if a new best validation_loss is achieved.
        model_file_path = self.model_dir
        self.checkpoint_callback = CustomModelCheckpoint(
            filepath=model_file_path,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )

        # Stop the training if the validation loss increases for
        # three epochs.
        self.es_callback = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=20,
            verbose=True,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        # Log all training data in a format suitable
        # for TensorBoard anaylsis.
        self.tb_callback = callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            profile_batch=0,
        )

        # Log the results of each epoch in a csv file.
        csv_file = os.path.join(self.log_dir, "log.csv")
        self.csv_callback = callbacks.CSVLogger(
            filename=csv_file, separator=",", append=True
        )

        # Update saved hyperparameters JSON file.
        self.hp_callback = HyperParamCallback(self)

    def make_dirs(self):
        """
        Make folder structure unless it already exists.

        Ensure that the folders are properly created.

        Returns
        -------
        None.

        """
        model_split = str(self.model_dir.split("runs")[1])
        log_split = str(self.log_dir.split("runs")[1])
        fig_split = str(self.fig_dir.split("runs")[1])

        if os.path.isdir(self.model_dir) is False:
            os.makedirs(self.model_dir)
            if os.path.isdir(self.model_dir):
                print("Model folder created at " + model_split)
        else:
            print("Model folder was already at " + model_split)

        if os.path.isdir(self.log_dir) is False:
            os.makedirs(self.log_dir)
            if os.path.isdir(self.log_dir):
                print("Logs folder created at " + log_split)
        else:
            print("Logs folder was already at " + log_split)

        if os.path.isdir(self.fig_dir) is False:
            os.makedirs(self.fig_dir)
            if os.path.isdir(self.fig_dir):
                print("Figures folder created at " + fig_split)
        else:
            print("Figures folder was already at " + fig_split)

    def activate_cbs(
        self,
        checkpoint=True,
        early_stopping=False,
        tb_log=False,
        csv_log=True,
        hyperparam_log=True,
        **cb_kwargs,
    ):
        """
        Activate the callbacks and store the active ones in a list.

        Method to be used before training.

        Parameters
        ----------
        checkpoint : boolean, optional
            Saving of best model during training.
            Should always be active
            The default is True.
        early_stopping : boolean, optional
            Implements early stopping.
            The default is False.
        tb_log : boolean, optional
            Initiates TensorBoard logging. The default is False.
        csv_log : boolean, optional
            Saving of history to CSV.
            Should always be active.
            The default is True.
        hyperparam_log : boolean, optional
            Saving of hyperparameters to JSON.
            Should always be active.
            The default is True.
        **cb_kwargs: dict
            Kwargs to pass to the callbacks.

        Returns
        -------
        None.

        """
        self.active_cbs = []

        if checkpoint:
            self.active_cbs.append(self.checkpoint_callback)
        if early_stopping:
            if "es_patience" in cb_kwargs.keys():
                self.es_callback.patience = cb_kwargs["es_patience"]
            self.active_cbs.append(self.es_callback)
        if tb_log:
            self.active_cbs.append(self.tb_callback)
        if csv_log:
            self.active_cbs.append(self.csv_callback)
        if hyperparam_log:
            self.active_cbs.append(self.hp_callback)

    def _count_epochs_trained(self):
        """
        Count the numbers of previously trained epochs.

        Seaches the csv log file.

        Returns
        -------
        epochs : int
            No. of epochs for which the model was trained.

        """
        csv_file = os.path.join(self.log_dir, "log.csv")
        epochs = 0

        try:
            with open(csv_file, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for _ in reader:
                    epochs += 1
        except FileNotFoundError:
            pass

        return epochs

    def _get_total_history(self):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing the previous training history.

        """
        csv_file = os.path.join(self.log_dir, "log.csv")

        history = {}
        try:
            with open(csv_file, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for key, item in row.items():
                        if key not in history.keys():
                            history[key] = []
                        history[key].append(float(item))
        except FileNotFoundError:
            pass

        return history

    def _purge_history(self):
        """
        Deletes the training history.
        Useful for resetting the classifier.
        """
        try:
            del (self.history, self.last_training)
        except AttributeError:
            pass

        self.update_saved_hyperparams({"epochs_trained": 0})

        for root, _, files in os.walk(self.log_dir):
            if root != self.log_dir:
                shutil.rmtree(root)
            else:
                for name in files:
                    if name != "hyperparameters.json":
                        os.unlink(os.path.join(root, name))

    def save_hyperparams(self):
        """
        Save all saved hyperparameters to a JSON file in the log_dir.

        Returns
        -------
        None.

        """
        hyperparam_file_name = os.path.join(self.log_dir, "hyperparameters.json")

        with open(hyperparam_file_name, "w", encoding="utf-8") as json_file:
            json.dump(
                self.hyperparams,
                json_file,
                ensure_ascii=False,
                indent=4,
            )

    def update_saved_hyperparams(self, new_params):
        """
        Update already saved hyperparameters with data from a new dict.

        Parameters
        ----------
        new_params : dict
            Dictionary with new hyperparameters.

        Returns
        -------
        None.

        """
        for key in new_params:
            self.hyperparams[key] = new_params[key]
        self.save_hyperparams()


class HyperParamCallback(callbacks.Callback):
    """A keras Callback saves the hyperparameter in a logging object."""

    def __init__(self, logging):
        """
        Initiate the logging object.

        Parameters
        ----------
        logging : ExperimentLogging
            Logging object with hyperparameter dictionary.

        Returns
        -------
        None.

        """
        self.logging = logging

    def on_epoch_end(self, epoch, logs=None):
        """
        Update the epochs_trained parameter in the logging.

        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : dict, optional
             Dict, metric results for this training epoch, and for the
             validation epoch if validation is performed.

        Returns
        -------
        None.

        """
        epoch_param = {"epochs_trained": self.logging._count_epochs_trained()}
        self.logging.update_saved_hyperparams(epoch_param)


class CustomModelCheckpoint(callbacks.ModelCheckpoint):
    """Extends the ModelCheckpoint class from Keras."""

    def _save_model(self, epoch, batch, logs):
        """
        Saves the model.

        Saves the model to JSON and the weights to HDF5 as well.

        Parameters
        ----------
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
            is set to `epoch`
            logs: the `logs` dict passed in to `on_batch_end` or
            `on_epoch_end`.
        """
        from tensorflow.python.keras.utils import tf_utils
        from tensorflow.python.platform import tf_logging as logging

        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning(
                            "Can save best model only with %s available, " "skipping.",
                            self.monitor,
                        )
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print(
                                    "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                    " saving model to %s"
                                    % (
                                        epoch + 1,
                                        self.monitor,
                                        self.best,
                                        current,
                                        filepath,
                                    )
                                )
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath,
                                    overwrite=True,
                                    options=self._options,
                                )
                            else:
                                with open(
                                    os.path.join(filepath, "model.json"),
                                    "w",
                                    encoding="utf-8",
                                ) as json_file:
                                    json_file.write(self.model.to_json())
                                self.model.save_weights(
                                    os.path.join(filepath, "weights.h5")
                                )
                                self.model.save(
                                    filepath,
                                    overwrite=True,
                                    options=self._options,
                                )
                        else:
                            if self.verbose > 0:
                                print(
                                    "\nEpoch %05d: %s did not improve from %0.5f"
                                    % (
                                        epoch + 1,
                                        self.monitor,
                                        self.best,
                                    )
                                )
                else:
                    if self.verbose > 0:
                        print(
                            "\nEpoch %05d: saving model to %s" % (epoch + 1, filepath)
                        )
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath,
                            overwrite=True,
                            options=self._options,
                        )
                    else:
                        with open(
                            os.path.join(filepath, "model.json"),
                            "w",
                            encoding="utf-8",
                        ) as json_file:
                            json_file.write(self.model.to_json())
                        self.model.save_weights(os.path.join(filepath, "weights.h5"))
                        self.model.save(
                            filepath,
                            overwrite=True,
                            options=self._options,
                        )

                self._maybe_remove_file()
            except IsADirectoryError as exc:  # h5py 3.x
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: {filepath}"
                ) from exc
            except IOError as exc:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if "is a directory" in str(exc.args[0]).lower():
                    raise IOError(
                        "Please specify a non-directory filepath for "
                        "ModelCheckpoint. Filepath used is an existing "
                        f"directory: f{filepath}"
                    ) from exc
                # Re-throw the error for any other causes.
                raise exc
