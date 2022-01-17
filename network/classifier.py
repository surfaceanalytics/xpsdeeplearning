# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:55:44 202.

@author: pielsticker
"""

import os
import pickle
import numpy as np
import json
from matplotlib import pyplot as plt

# Disable tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from .data_handling import DataHandler
from .exp_logging import ExperimentLogging
from .utils import WeightDistributions

from . import models

#%%
class Classifier:
    """Class for training and testing a Keras model."""

    def __init__(
        self,
        time,
        exp_name="",
        task="regression",
        intensity_only=True,
        labels=[],
    ):
        """
        Initialize folder structure and an empty model.

        Parameters
        ----------
        time : str
            Time when the experiment is initiated.
        exp_name : str, optional
            Name of the experiment. Should include sufficient information
            for distinguishing runs on different data on the same day.
            The default is ''.
        task : str, optional
            Task to perform. Either 'classification', 
            "multi_class_detection or 'regression'.
            The default is 'regression'.
        intensity_only : boolean, optional
            If True, then only the intensity scale is loaded into X.
            If False, the intensity and the BE scale is loaded into X.
            The default is True.
        labels : list
            List of strings of the labels of the XPS spectra.
            Can also be loaded from file.
            Example: 
                labels = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']
            Can be empty if the labels are stored in the data file.

        Returns
        -------
        None.

        """
        self.time = time
        self.exp_name = exp_name
        self.task = task
        self.intensity_only = intensity_only

        self.model = Model()

        self.datahandler = DataHandler(intensity_only)
        self.datahandler.labels = labels
        self.datahandler.num_classes = len(self.datahandler.labels)

        dir_name = self.time + "_" + self.exp_name
        self.logging = ExperimentLogging(dir_name)
        self.logging.make_dirs()

        self.logging.hyperparams = {
            "time": self.time,
            "exp_name": self.exp_name,
            "task": self.task,
            "intensity_only": self.intensity_only,
            "epochs_trained": 0,
        }
        self.logging.save_hyperparams()

    def summary(self):
        """
        Print a summary of the keras Model in self.model.
        
        Save the string value of the summary to the self.model_summary
        attribute.

        Returns
        -------
        None.

        """

        print(self.model.summary())

        # Save model summary to a string
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)

    def save_and_print_model_image(self):
        """
        Plot the model using the plot_model method from keras.
        
        Save the image to a file in the figure directory.

        Returns
        -------
        None.

        """
        fig_file_name = os.path.join(self.logging.fig_dir, "model.png")
        plot_model(
            self.model,
            to_file=fig_file_name,
            rankdir="LR",
            show_shapes=True,
            show_layer_names=True,
        )
        model_plot = plt.imread(fig_file_name)
        fig, ax = plt.subplots(figsize=(18, 2))
        ax.imshow(model_plot, interpolation="nearest")
        plt.tight_layout()
        plt.show()

    def train(
        self,
        epochs,
        batch_size,
        checkpoint=True,
        early_stopping=False,
        tb_log=False,
        csv_log=True,
        hyperparam_log=True,
        cb_parameters={},
        verbose=1,
        new_learning_rate=None,
        *args,
        **kwargs,
    ):
        """
        Train the keras model.
        
        Implements various callbacks during
        the training. Also checks the csv_log for previous training and
        appends accordingly.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to be trained. The default is 200.
        batch_size : int, optional
            Batch size for stochastic optimization.
            The default is 32.
        checkpoint : bool, optional
            Determines if the model is saved when the val_loss is lowest.
            The default is True.
        early_stopping : bool, optional
            If early_stopping, the training is stopped when the val_loss
            increases for three straight epochs.
            The default is False.
        tb_log : bool, optional
            Controls the logging in Tensorboard.
            The default is False.
        csv_log : bool, optional
            If True, the data for each epoch is logged in a CSV file.
            Needs to be True if later retraining is planned.
            The default is True.
        hyperparam_log : bool, optional
            If True, the hyperparameters of the classifier are saved
            in a CSV file.
            Needs to be True if later retraining is planned.
            The default is True.
        verbose : int, optional
            Controls the verbose parameter of the keras output.
            If 1, a progress log bar is shown in the output. 
            The default is 1.
        new_learning_rate : float, optional
            In case of retraining, the learning rate of the optimizer can
            be overwritten.
            The default is None.

        Returns
        -------
        self.logging.history: dict
            Dictionary containing the training results for each epoch.

        """
        self.logging.hyperparams["batch_size"] = batch_size
        # In case of refitting, get the new epoch start point.
        epochs_trained = self.logging._count_epochs_trained()

        if new_learning_rate is not None:
            # Overwrite the previus optimizer learning rate using the
            # backend of keras.
            K.set_value(self.model.optimizer.learning_rate, new_learning_rate)
            print(
                "New learning rate: " + str(K.eval(self.model.optimizer.lr))
            )

        # Activate callbacks in experiment logging.
        self.logging.activate_cbs(
            checkpoint=checkpoint,
            early_stopping=early_stopping,
            tb_log=tb_log,
            csv_log=csv_log,
            hyperparam_log=hyperparam_log,
            **cb_parameters,
        )

        # Update json file with hyperparameters.
        train_params = {
            "model_summary": self.model_summary,
            "epochs_trained": epochs_trained,
            "batch_size": self.logging.hyperparams["batch_size"],
            "loss": type(self.model.loss).__name__,
            "optimizer": type(self.model.optimizer).__name__,
            "learning_rate": str(K.eval(self.model.optimizer.lr)),
        }
        self.logging.update_saved_hyperparams(train_params)

        try:
            # Train the model and store the previous and the new
            # results in the history attribute.
            training = self.model.fit(
                self.datahandler.X_train,
                self.datahandler.y_train,
                validation_data=(
                    self.datahandler.X_val,
                    self.datahandler.y_val,
                ),
                epochs=epochs + epochs_trained,
                batch_size=self.logging.hyperparams["batch_size"],
                initial_epoch=epochs_trained,
                verbose=verbose,
                callbacks=self.logging.active_cbs,
                *args,
                **kwargs,
            )

            self.logging.last_training = training.history
            self.logging.history = self.logging._get_total_history()
            print("Training done!")

        except KeyboardInterrupt:
            # Save the model and the history in case of interruption.
            print("Training interrupted!")
            if all(
                checkpoint
                not in [type(cb).__name__ for cb in self.logging.active_cbs]
                for checkpoint in ("ModelCheckpoint", "CustomModelCheckpoint")
            ):
                self.save_model()
            self.logging.history = self.logging._get_total_history()

        epoch_param = {"epochs_trained": self.logging._count_epochs_trained()}
        self.logging.update_saved_hyperparams(epoch_param)

        return self.logging.history

    def evaluate(self):
        """
        Evaluate the Model on the test data.

        Returns
        -------
        score: dict or float
            If the accuracy is calculated, both the test loss and the
            test accuracy are returned. If not, only the test loss 
            is returned.
        """
        score = self.model.evaluate(
            self.datahandler.X_test,
            self.datahandler.y_test,
            batch_size=self.logging.hyperparams["batch_size"],
            verbose=True,
        )
        print("Evaluation done! \n")

        try:
            self.datahandler.test_loss, self.datahandler.test_accuracy = (
                score[0],
                score[1],
            )
        except TypeError:
            self.datahandler.test_loss = score

        return score

    def predict(self, verbose=True):
        """
        Predict the labels on both the train and tests sets.

        Parameters
        ----------
        verbose : bool, optional
            If True, show the progress bar during prediction.
            The default is True.

        Returns
        -------
        pred_train : ndarray
            Array containing the predictions on the training set.
        pred_test : ndarray
            Array containing the predictions on the test set.

        """
        self.datahandler.pred_train = self.model.predict(
            self.datahandler.X_train, verbose=verbose
        )
        self.datahandler.pred_test = self.model.predict(
            self.datahandler.X_test, verbose=verbose
        )

        if verbose:
            print("Prediction done!")

        return self.datahandler.pred_train, self.datahandler.pred_test

    def predict_classes(self):
        """
        Predict the labels of all spectra in the training/test sets.
        
        This is done by checking which class has the maximum associated
        prediction value.

        Returns
        -------
        pred_train_classes : ndarray
            Array with the predicted classes on the training set.
        pred_test_classes : ndarray
            Array with the predicted classes on the test set.

        """
        if self.task == "regression":
            print(
                "Regression was chosen as task. "
                + "No prediction of classes possible!"
            )
        if self.task == "multi_class_detection":
            self.datahandler.pred_train_classes = []
            self.datahandler.pred_test_classes = []
                           
            for i, pred in enumerate(clf.datahandler.pred_train):
                arg_max = list(np.where(pred > 0.05)[0])
                classes = [clf.datahandler.labels[arg] for arg in arg_max]
                self.datahandler.pred_train_classes.append(classes)
 
            for i, pred in enumerate(clf.datahandler.pred_test):
                arg_max = list(np.where(pred > 0.05)[0])
                classes = [clf.datahandler.labels[arg] for arg in arg_max]
                self.datahandler.pred_test_classes.append(classes)
            
            print("Class prediction done!") 
            
            return (
                self.datahandler.pred_train_classes,
                self.datahandler.pred_test_classes,
            )
        
        else:
            pred_train_classes = []
            pred_test_classes = []

            for i, pred in enumerate(clf.datahandler.pred_train):
                argmax_class = np.argmax(pred, axis=0)
                pred_train_classes.append(
                    self.datahandler.labels[argmax_class]
                )

            for i, pred in enumerate(clf.datahandler.pred_test):
                argmax_class = np.argmax(pred, axis=0)
                pred_test_classes.append(
                    self.datahandler.labels[argmax_class]
                )

            self.datahandler.pred_train_classes = np.array(
                pred_train_classes
            ).reshape(-1, 1)
            self.datahandler.pred_test_classes = np.array(
                pred_test_classes
            ).reshape(-1, 1)

            print("Class prediction done!")

            return (
                self.datahandler.pred_train_classes,
                self.datahandler.pred_test_classes,
            )

    def save_model(self):
        """
        Save the model to the model directory.
        
        The model is saved both as a SavedModel object from Keras as 
        well as a JSON file containing the model parameters and the 
        weights serialized to HDF5.

        Returns
        -------
        None.

        """
        model_file_name = os.path.join(self.logging.model_dir, "model.json")
        weights_file_name = os.path.join(self.logging.model_dir, "weights.h5")

        model_json = self.model.to_json()
        with open(model_file_name, "w", encoding="utf-8") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_file_name)
        self.model.save(self.logging.model_dir)

        print("Saved model to disk.")

    def load_model(
        self, model_path=None, drop_last_layers=None, compile_model=True
    ):
        """
        Reload the model from file.

        Parameters
        ----------
        model_path : str, optional
            If model_path != None, then the model is loaded from a
            filepath.
            If None, the model is loaded from self.logging.model_dir.
            The default is None.
        drop_last_layers : int, optional
            No. of layers to be dropped during the loading. Helpful for
            transfer learning.
            The default is None.
        compile_model: bool, optional
            Whether to compile the model after loading, using the
            saved optimizer and loss.            

        Returns
        -------
        None.

        """
        self.model = Model()
        if model_path is not None:
            file_name = model_path
        else:
            file_name = self.logging.model_dir

        # Add the current model to the custom_objects dict. This is
        # done to ensure the load_model method from Keras works
        # properly.
        import inspect

        custom_objects = {}
        custom_objects[str(type(self.model).__name__)] = self.model.__class__
        for name, obj in inspect.getmembers(models):
            if inspect.isclass(obj):
                if obj.__module__.startswith(
                    "xpsdeeplearning.network.models"
                ):
                    custom_objects[obj.__module__ + "." + obj.__name__] = obj

        # Load from file.
        loaded_model = load_model(file_name, custom_objects=custom_objects)
        optimizer = loaded_model.optimizer
        loss = loaded_model.loss
        metrics = [
            metric.name
            for metric in loaded_model.metrics
            if metric.name != "loss"
        ]

        # Instantiate a new EmptyModel and implement it with the loaded
        # parameters. Needed for dropping the layers while keeping all
        # parameters.
        self.model = models.EmptyModel(
            inputs=loaded_model.input,
            outputs=loaded_model.layers[-1].output,
            inputshape=self.datahandler.input_shape,
            num_classes=self.datahandler.num_classes,
            no_of_inputs=loaded_model._serialized_attributes["metadata"][
                "config"
            ]["no_of_inputs"],
            name="Loaded_Model",
        )

        print("Loaded model from disk.")

        if drop_last_layers is not None:
            # Remove the last layers and stored the other layers in a
            # new EmptyModel object.
            no_of_drop_layers = drop_last_layers + 1

            new_model = models.EmptyModel(
                inputs=self.model.input,
                outputs=self.model.layers[-no_of_drop_layers].output,
                inputshape=self.datahandler.input_shape,
                num_classes=self.datahandler.num_classes,
                no_of_inputs=self.model.no_of_inputs,
                name="Changed_Model",
            )

            self.model = new_model

            if drop_last_layers == 0:
                print("No layers were dropped.\n")

            elif drop_last_layers == 1:
                print("The last layer was dropped.\n")

            else:
                print(
                    "The last {} layers were dropped.\n".format(
                        str(drop_last_layers)
                    )
                )

        if compile_model:
            self.model.compile(
                optimizer=optimizer, loss=loss, metrics=metrics
            )

    def load_data_preprocess(
        self,
        input_filepath,
        no_of_examples,
        train_test_split,
        train_val_split,
    ):
        """
        Load the data.
        
        Utilizes load_data_preprocess method from DataHandler. 
        Stores all loading parameters in the 
        hyperparameters.json file.

        Parameters
        ----------
        input_filepath : str
            Filepath of the .
        no_of_examples : int
            Number of samples from the input file.
        train_test_split : float
            Split percentage between train+val and test set.
            Typically ~ 0.2.
        train_val_split : float
            Split percentage between train and val set.
            Typically ~ 0.2.

        Returns
        -------
        loaded_data : tuple
            Return the output of the load_data_preprocess method
            from DataHandler.

        """
        loaded_data = self.datahandler.load_data_preprocess(
            input_filepath, no_of_examples, train_test_split, train_val_split
        )

        energy_range = [
            np.min(self.datahandler.energies),
            np.max(self.datahandler.energies),
            np.round(
                self.datahandler.energies[0] - self.datahandler.energies[1], 2
            ),
        ]

        data_params = {
            "input_filepath": self.datahandler.input_filepath,
            "train_test_split": self.datahandler.train_test_split,
            "train_val_split": self.datahandler.train_val_split,
            "labels": self.datahandler.labels,
            "num_of_classes": self.datahandler.num_classes,
            "no_of_examples": self.datahandler.no_of_examples,
            "energy_range": energy_range,
            "No. of training samples": str(self.datahandler.X_train.shape[0]),
            "No. of validation samples": str(self.datahandler.X_val.shape[0]),
            "No. of test samples": str(self.datahandler.X_test.shape[0]),
            "Shape of each sample": str(self.datahandler.X_train.shape[1])
            + " features (X)"
            + " + "
            + str(self.datahandler.y_train.shape[1])
            + " labels (y)",
        }

        self.logging.update_saved_hyperparams(data_params)

        return loaded_data

    def plot_random(
        self, no_of_spectra, dataset="train", with_prediction=False
    ):
        """
        Plot random XPS spectra out of one of the data set.
        
        The labels and additional information are shown as texts on the
        plots.
        Utilizes the method of the same name in DataHandler.
        
        Parameters
        ----------
        no_of_spectra : int
            No. of plots to create.
        dataset : str, optional
            Either 'train', 'val', or 'test'.
            The default is 'train'.
        with_prediction : bool, optional
            If True, information about the predicted values are also 
            shown in the plot. 
            The default is False.

        Returns
        -------
        None.

        """
        if with_prediction:
            try:
                self.datahandler.plot_random(
                    no_of_spectra, dataset, with_prediction=True
                )
            except AttributeError:
                self.datahandler.calculate_losses(self.model.loss)
                self.datahandler.plot_random(
                    no_of_spectra, dataset, with_prediction=True
                )
        else:
            self.datahandler.plot_random(
                no_of_spectra, dataset, with_prediction
            )

    def show_worst_predictions(
        self, no_of_spectra, kind="all", threshold=0.0
    ):
        """
        Plot the spectra with the highest losses.
        
        Accepts a threshold parameter. If a threshold other than 0 is
        given, the spectra with losses right above this threshold are
        plotted.
        Utilizes the method of the same name in DataHandler.

        Parameters
        ----------
        no_of_spectra : int
            No. of spectra to plot.
        kind : str, optional
            Choice of sub set in test data.
            'all': all test data.
            'single': only test data with single species.
            'linear_comb': only test data with linear combination 
                           of  species.
            The default is 'all'.
        threshold : float
            Threshold value for loss.            

        Returns
        -------
        None.

        """
        try:
            self.datahandler.show_worst_predictions(
                no_of_spectra, kind, threshold
            )
        except AttributeError:
            self.datahandler.calculate_losses(self.model.loss)
            self.datahandler.show_worst_predictions(
                no_of_spectra, kind, threshold
            )

    def show_wrong_classification(self):
        """
        Plot all spectra in the test data with a wrong prediction.
        
        Utilizes the method of the same name in DataHandler.

        Returns
        -------
        None.

        """
        self.datahandler.show_wrong_classification()

    def plot_class_distribution(self):
        """
        Plot the class distribution of the data.
        
        Returns
        -------
        None.

        """
        self.datahandler.class_distribution.plot(self.datahandler.labels)

    def plot_weight_distribution(self, kind="posterior", to_file=True):

        bayesian_layers = [
            layer
            for layer in self.model.layers
            if ("Flipout" or "Reparameterization") in str(layer.__class__)
        ]

        wd = WeightDistributions(
            bayesian_layers, fig_dir=self.logging.fig_dir
        )

        if kind == "prior":
            fig = wd.plot_weight_priors(to_file=to_file)
        elif kind == "posterior":
            fig = wd.plot_weight_posteriors(to_file=to_file)

        if to_file:
            epoch = self.logging.hyperparams["epochs_trained"]

            filename = os.path.join(
                self.logging.fig_dir,
                f"weights_{kind}_after_epoch_{epoch}.png",
            )

            fig.savefig(filename)
            print("Saved to {}".format(filename))

    def predict_probabilistic(self, dataset="test", no_of_predictions=100):

        X, y = self.datahandler._select_dataset(dataset_name=dataset)

        prob_pred = np.array(
            [self.model.predict(X) for i in range(no_of_predictions)]
        ).transpose(1, 0, 2)

        if dataset == "train":
            self.datahandler.prob_pred_train = prob_pred
            self.datahandler.pred_train = np.mean(prob_pred, axis=1)

        elif dataset == "test":
            self.datahandler.prob_pred_test = prob_pred
            self.datahandler.pred_test = np.mean(prob_pred, axis=1)

        return prob_pred

    def plot_prob_predictions(
        self, dataset="test", no_of_spectra=10, to_file=True
    ):

        prob_preds = self.predict_probabilistic(
            dataset=dataset, no_of_predictions=100
        )

        fig = self.datahandler.plot_prob_predictions(
            prob_preds=prob_preds,
            dataset=dataset,
            no_of_spectra=no_of_spectra,
        )

        if to_file:
            epoch = self.logging.hyperparams["epochs_trained"]

            filename = os.path.join(
                self.logging.fig_dir,
                f"probabilistic_predictions_after_epoch_{epoch}.png",
            )

            fig.savefig(filename)
            print("Saved to {}".format(filename))

    def pickle_results(self):
        """
        Pickle and save all outputs.

        Returns
        -------
        None.

        """
        filename = os.path.join(self.logging.log_dir, "results.pkl")

        key_list = [
            "y_train",
            "y_test",
            "pred_train",
            "pred_test",
            "test_loss",
            "class_distribution",
        ]
        if self.task != "regression":
            key_list.extend(
                ["test_accuracy", "pred_train_classes", "pred_test_classes"]
            )

        pickle_data = {}
        for key in key_list:
            try:
                if key == "class_distribution":
                    cd = getattr(self.datahandler, key)
                    pickle_data[key] = cd.cd
                else:
                    try:
                        pickle_data[key] = getattr(self.datahandler, key)
                    except AttributeError:
                        print(f"'DataHandler' object has no attribute {key}.")
            except KeyError:
                pass

        with open(filename, "wb") as pickle_file:
            pickle.dump(
                pickle_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL
            )

        print("Saved results to file.")

    def purge_history(self):
        self.logging._purge_history()
        print("Training history was deleted.")


def restore_clf_from_logs(runpath):
    """
    Restore a Classifier object.
    
    The classifier is created from the logs saved in a previous 
    experiment.

    Parameters
    ----------
    runpath : str
        Path where the files from a previous experiment are located.

    Returns
    -------
    clf : Classifier
        Classifier object.

    """
    hyperparam_file_name = os.path.join(
        runpath, "logs", "hyperparameters.json"
    )

    with open(hyperparam_file_name, "r") as json_file:
        hyperparams = json.load(json_file)
        exp_name = hyperparams["exp_name"]
        time = hyperparams["time"]
        task = hyperparams["task"]
        intensity_only = hyperparams["intensity_only"]

    clf = Classifier(
        time=time, exp_name=exp_name, task=task, intensity_only=intensity_only
    )
    clf.logging.hyperparams = hyperparams
    clf.logging.history = clf.logging._get_total_history()
    clf.logging.save_hyperparams()

    print("Recovered classifier from file.")

    return clf


#%%
if __name__ == "__main__":
    np.random.seed(502)
    time = "20210218_09h21m"
    exp_name = "Fe_4_classes_linear_comb_new_noise_small_resnet"

    clf = Classifier(time=time, exp_name=exp_name)
    input_filepath = r"C:\Users\pielsticker\Simulations\20210222_Fe_linear_combination_small_gas_phase.h5"
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 1000

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
    ) = clf.datahandler.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )
