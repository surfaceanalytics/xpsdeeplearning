# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:29:41 2021.

@author: pielsticker
"""

import numpy as np
import h5py
from sklearn.utils import shuffle

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .utils import ClassDistribution, SpectraPlot

#%%
class DataHandler:
    """Class for data treatment during an experiment in tensorflow."""

    def __init__(self, intensity_only=True):
        """
        Initialize intenstiy_only parameter.

        Parameters
        ----------
        intensity_only : boolean, optional
            If True, then only the intensity scale is loaded into X.
            If False, the intensity and the BE scale is loaded into X.
            The default is True.

        Returns
        -------
        None.

        """
        self.intensity_only = intensity_only

    def load_data_preprocess(
        self,
        input_filepath,
        no_of_examples,
        train_test_split,
        train_val_split,
    ):
        """
        Load the data from an HDF5 file and preprocess it.

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
        self.X_train : ndarray
            Training features. Shape: 3d Numpy array.
        self.X_val : ndarray
            Validation features.
        self.X_test : ndarray
            Test features.
        self.y_train : ndarray. Shape: 2d Numpy array.
            Training labels.
        self.y_val : ndarray
            Validation labels.
        self.y_test : ndarray
            Test labels.

        Optionally, the method can also return more information about
        the data set:
            self.sim_values_train,
            self.sim_values_val,
            self.sim_values_test : dicts
                Dictionary containing information about the parameters
                used during the artificical constructuon of the dataset.
                Keys : 'shiftx', 'noise', 'FWHM',
                       'scatterer', 'distance', 'pressure'
                Split in the same way as the features and labels.

             self.names_train,
             self.names_val,
             self.names_test : ndarrays
                 Arrays of the spectra names associated  with each X,y
                 pairing in the data set. Typical for measured spectra.
                 Split in the same way as the features and labels.
        """
        self.input_filepath = input_filepath
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.no_of_examples = no_of_examples

        with h5py.File(input_filepath, "r") as hf:
            try:
                self.energies = hf["energies"][:]
            except KeyError:
                self.energies = np.flip(np.arange(694, 750.05, 0.05))
                print(
                    "The data set did have not an energy scale. "
                    + "Default (Fe) was assumed."
                )
            try:
                try:
                    self.labels = [
                        label.decode("utf-8") for label in hf["labels"][:]
                    ]
                except AttributeError:
                    self.labels = [str(label) for label in hf["labels"][:]]
                self.num_classes = len(self.labels)
            except KeyError:
                print(
                    "The data set did not contain any labels. "
                    + "The label list is empty."
                )

            dataset_size = hf["X"].shape[0]
            # Randomly choose a subset of the whole data set.
            try:
                r = np.random.randint(0, dataset_size - self.no_of_examples)
            except ValueError as e:
                error_msg = (
                    "There are not enough spectra in this data set. "
                    + "Please choose a value of less than {0} ".format(
                        dataset_size - 1
                    )
                    + "for no_of_examples."
                )
                raise type(e)(error_msg)

            X = hf["X"][r : r + self.no_of_examples, :, :]
            X = X.astype(np.float)
            y = hf["y"][r : r + self.no_of_examples, :]

            if not self.intensity_only:
                new_energies = np.tile(
                    np.reshape(np.array(self.energies), (-1, 1)),
                    (X.shape[0], 1, 1),
                )
                X = np.dstack((new_energies, X))

            # Check if the data set was artificially created.
            if "shiftx" in hf.keys():
                shift_x = hf["shiftx"][r : r + self.no_of_examples, :]
                noise = hf["noise"][r : r + self.no_of_examples, :]
                fwhm = hf["FWHM"][r : r + self.no_of_examples, :]

                if "scatterer" in hf.keys():
                    # If the data set was artificially created, check
                    # if scattering in a gas phase was simulated.
                    scatterer = hf["scatterer"][
                        r : r + self.no_of_examples, :
                    ]
                    distance = hf["distance"][r : r + self.no_of_examples, :]
                    pressure = hf["pressure"][r : r + self.no_of_examples, :]

                    (
                        self.X,
                        self.y,
                        shift_x,
                        noise,
                        fwhm,
                        scatterer,
                        distance,
                        pressure,
                    ) = shuffle(
                        X,
                        y,
                        shift_x,
                        noise,
                        fwhm,
                        scatterer,
                        distance,
                        pressure,
                    )

                    # Store all parameters of the simulations in a dict
                    sim_values = {
                        "shift_x": shift_x,
                        "noise": noise,
                        "fwhm": fwhm,
                        "scatterer": scatterer,
                        "distance": distance,
                        "pressure": pressure,
                    }
                    self.sim_values = sim_values

                else:
                    # Shuffle all arrays together
                    self.X, self.y, shift_x, noise, fwhm = shuffle(
                        X, y, shift_x, noise, fwhm
                    )
                    sim_values = {
                        "shift_x": shift_x,
                        "noise": noise,
                        "fwhm": fwhm,
                    }
                    self.sim_values = sim_values

                # Split into train, val and test sets
                (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                    self.sim_values_train,
                    self.sim_values_val,
                    self.sim_values_test,
                ) = self._split_test_val_train(
                    self.X, self.y, sim_values=self.sim_values
                )

                # Determine the shape of the training features,
                # needed for model building in Keras.
                self.input_shape = self.X_train.shape[1:]

                loaded_data = (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                    self.sim_values_train,
                    self.sim_values_val,
                    self.sim_values_test,
                )

            # Check if the spectra have associated names. Typical for
            # measured spectra.
            elif "names" in hf.keys():
                names_load_list = [
                    name[0].decode("utf-8")
                    for name in hf["names"][r : r + self.no_of_examples, :]
                ]
                names = np.reshape(np.array(names_load_list), (-1, 1))

                # Shuffle all arrays together
                self.X, self.y, self.names = shuffle(X, y, names)

                # Split into train, val and test sets
                (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                    self.names_train,
                    self.names_val,
                    self.names_test,
                ) = self._split_test_val_train(
                    self.X, self.y, names=self.names
                )

                # Determine the shape of the training features,
                # needed for model building in Keras.
                self.input_shape = (
                    self.X_train.shape[1],
                    self.X_train.shape[2],
                )

                loaded_data = (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                    self.names_train,
                    self.names_val,
                    self.names_test,
                )

            # If there are neither simulation values nor names in
            # the dataset, just load the X and y arrays.
            else:
                # Shuffle X and y together
                self.X, self.y = shuffle(X, y)
                # Split into train, val and test sets
                (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                ) = self._split_test_val_train(self.X, self.y)

                # Determine the shape of the training features,
                # needed for model building in Keras.
                self.input_shape = (
                    self.X_train.shape[1],
                    self.X_train.shape[2],
                )

                loaded_data = (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                )

        print("Data was loaded!")
        print("Total no. of samples: " + str(self.X.shape[0]))
        print("No. of training samples: " + str(self.X_train.shape[0]))
        print("No. of validation samples: " + str(self.X_val.shape[0]))
        print("No. of test samples: " + str(self.X_test.shape[0]))
        print(
            "Shape of each sample : "
            + str(self.X_train.shape[1])
            + " features (X)"
            + " + "
            + str(self.y_train.shape[1])
            + " labels (y)"
        )

        return loaded_data

    def _split_test_val_train(self, X, y, **kwargs):
        """
        Split multiple numpy arrays into train, val, and test sets.

        First, the whole data is split into the train+val and test sets
        according to the attribute self.train_test_split. Secondly.
        the train+val sets are further split into train and val sets
        according to the attribute self.train_val_split.

        Parameters
        ----------
        X : ndarray
            Features used as inputs for a Keras model. 3d array.
        y : TYPE
            Labels to be learned. 2d array.
        **kwargs : str
            Possible keywords:
                'sim_values', 'names'.

        Returns
        -------
        For each input array, three output arras are returned.
        E.g. for input X, the returns are X_train, X_val, X_test.
        """
        # First split into train+val and test sets
        no_of_train_val = int((1 - self.train_test_split) * X.shape[0])

        X_train_val = X[:no_of_train_val, :, :]
        X_test = X[no_of_train_val:, :, :]
        y_train_val = y[:no_of_train_val, :]
        y_test = y[no_of_train_val:, :]

        # Then create val subset from train set
        no_of_train = int((1 - self.train_val_split) * X_train_val.shape[0])

        X_train = X_train_val[:no_of_train, :, :]
        X_val = X_train_val[no_of_train:, :, :]
        y_train = y_train_val[:no_of_train, :]
        y_val = y_train_val[no_of_train:, :]

        if "sim_values" in kwargs.keys():
            # Also split the arrays in the 'sim_values' dictionary.
            sim_values = kwargs["sim_values"]
            shift_x = sim_values["shift_x"]
            noise = sim_values["noise"]
            fwhm = sim_values["fwhm"]

            shift_x_train_val = shift_x[:no_of_train_val, :]
            shift_x_test = shift_x[no_of_train_val:, :]
            noise_train_val = noise[:no_of_train_val, :]
            noise_test = noise[no_of_train_val:, :]
            fwhm_train_val = fwhm[:no_of_train_val, :]
            fwhm_test = fwhm[no_of_train_val:, :]

            shift_x_train = shift_x_train_val[:no_of_train, :]
            shift_x_val = shift_x_train_val[no_of_train:, :]
            noise_train = noise_train_val[:no_of_train, :]
            noise_val = noise_train_val[no_of_train:, :]
            fwhm_train = fwhm_train_val[:no_of_train, :]
            fwhm_val = fwhm_train_val[no_of_train:, :]

            sim_values_train = {
                "shift_x": shift_x_train,
                "noise": noise_train,
                "fwhm": fwhm_train,
            }
            sim_values_val = {
                "shift_x": shift_x_val,
                "noise": noise_val,
                "fwhm": fwhm_val,
            }
            sim_values_test = {
                "shift_x": shift_x_test,
                "noise": noise_test,
                "fwhm": fwhm_test,
            }

            if "scatterer" in sim_values.keys():
                # Also split the scatterer, distance, and pressure
                # arrays if they are present in the in the 'sim_values'
                # dictionary.
                scatterer = sim_values["scatterer"]
                distance = sim_values["distance"]
                pressure = sim_values["pressure"]

                scatterer_train_val = scatterer[:no_of_train_val, :]
                scatterer_test = scatterer[no_of_train_val:, :]
                distance_train_val = distance[:no_of_train_val, :]
                distance_test = distance[no_of_train_val:, :]
                pressure_train_val = pressure[:no_of_train_val, :]
                pressure_test = pressure[no_of_train_val:, :]

                scatterer_train = scatterer_train_val[:no_of_train, :]
                scatterer_val = scatterer_train_val[no_of_train:, :]
                distance_train = distance_train_val[:no_of_train, :]
                distance_val = distance_train_val[no_of_train:, :]
                pressure_train = pressure_train_val[:no_of_train, :]
                pressure_val = pressure_train_val[no_of_train:, :]

                sim_values_train["scatterer"] = scatterer_train
                sim_values_train["distance"] = distance_train
                sim_values_train["pressure"] = pressure_train

                sim_values_val["scatterer"] = scatterer_val
                sim_values_val["distance"] = distance_val
                sim_values_val["pressure"] = pressure_val

                sim_values_test["scatterer"] = scatterer_test
                sim_values_test["distance"] = distance_test
                sim_values_test["pressure"] = pressure_test

            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                sim_values_train,
                sim_values_val,
                sim_values_test,
            )

        if "names" in kwargs.keys():
            # Also split the names array.
            names = kwargs["names"]
            names_train_val = names[:no_of_train_val, :]
            names_test = names[no_of_train_val:, :]

            names_train = names_train_val[:no_of_train, :]
            names_val = names_train_val[no_of_train:, :]

            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                names_train,
                names_val,
                names_test,
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _only_keep_classification_data(self):
        """
        Keep only data with just one species, i.e. with one label of 1.0
        and the rest of the labels = 0.0.
        """
        indices = np.where(self.y == 0.0)[0]

        self.X, self.y = self.X[indices], self.y[indices]

        if hasattr(self, "sim_values"):
            new_sim_values = {}

            for key, sim_arrays in self.sim_values.items():
                new_sim_values[key] = sim_arrays[indices]

            self.sim_values = new_sim_values

            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.sim_values_train,
                self.sim_values_val,
                self.sim_values_test,
            ) = self._split_test_val_train(
                self.X, self.y, sim_values=self.sim_values
            )

            loaded_data = (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.sim_values_train,
                self.sim_values_val,
                self.sim_values_test,
            )

        elif hasattr(self, "names"):
            print("Hi")
            self.names = [self.names[i] for i in indices]

            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.names_train,
                self.names_val,
                self.names_test,
            ) = self._split_test_val_train(self.X, self.y, names=self.names)

            loaded_data = (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.names_train,
                self.names_val,
                self.names_test,
            )

        else:
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
            ) = self._split_test_val_train(self.X, self.y)

            loaded_data = (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
            )
        print(
            f"Only spectra with one species were left in the data set! Test/val/train splits were kept."
        )
        print(f"Remaining no. of training examples: {self.y_train.shape[0]}")
        print(f"Remaining no. of val examples: {self.y_val.shape[0]}")
        print(f"Remaining no. of test examples: {self.y_test.shape[0]}")

        return loaded_data

    def check_class_distribution(self, task):
        """
        Generate a Class Distribution object based on a given task.

        Parameters
        ----------
        task : str
            If task == 'regression', an average distribution is
            calculated.
            If task == 'classification' or 'multi_class_detection',
            the distribution of the labels across the different data
            sets is calculated.

        Returns
        -------
        dict
            Dictionary containing the class distribution.

        """
        data_list = [self.y, self.y_train, self.y_val, self.y_test]
        self.class_distribution = ClassDistribution(task, data_list)
        return self.class_distribution.cd

    def calculate_losses(self, loss_func):
        """
        Calculate losses for train and test data.

        Parameters
        ----------
        loss_func : keras.losses.Loss
            A keras loss function.

        Returns
        -------
        None.

        """
        print("Calculating loss for each example...")
        self.losses_train = [
            loss_func(self.y_train[i], self.pred_train[i]).numpy()
            for i in range(self.y_train.shape[0])
        ]

        self.losses_test = [
            loss_func(self.y_test[i], self.pred_test[i]).numpy()
            for i in range(self.y_test.shape[0])
        ]
        print("Done!")

    def plot_spectra(self, no_of_spectra, dataset, indices, with_prediction):
        """
        Generate spectra plot for a given data set.

        Parameters
        ----------
        no_of_spectra : int
            No. of plots to create.
        dataset : str
            Either 'train', 'val', or 'test'.
            The default is 'train'.
        indices: list
            List
        with_prediction : bool, optional
            If True, information about the predicted values are also
            shown in the plot.
            The default is False.

        Returns
        -------
        None.

        """
        data = []
        texts = []

        X, y = self._select_dataset(dataset)

        for i in range(no_of_spectra):
            index = indices[i]
            if self.intensity_only:
                new_energies = np.reshape(np.array(self.energies), (-1, 1))
                data.append(np.hstack((new_energies, X[index])))
            else:
                data.append(X[index])

            text = self.write_text_for_spectrum(
                dataset=dataset,
                index=index,
                with_prediction=with_prediction,
            )

            texts.append(text)

        data = np.array(data)

        graphic = SpectraPlot(data=data, annots=texts)
        fig, axs = graphic.plot()

    def plot_random(self, no_of_spectra, dataset, with_prediction):
        """
        Plot random XPS spectra out of one of the data set.

        The labels and additional information are shown as texts on the
        plots.

        Parameters
        ----------
        no_of_spectra : int
            No. of plots to create.
        dataset : str
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
        X, y = self._select_dataset(dataset)

        indices = []
        for i in range(no_of_spectra):
            r = np.random.randint(0, X.shape[0])
            indices.append(r)

        self.plot_spectra(
            no_of_spectra=no_of_spectra,
            dataset=dataset,
            indices=indices,
            with_prediction=with_prediction,
        )

    def show_worst_predictions(
        self, no_of_spectra, kind="all", threshold=0.0
    ):
        """
        Plot the spectra with the highest losses.

        Accepts a threshold parameter. If a threshold other than 0 is
        given, the spectra with losses right above this threshold are
        plotted.

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
        X, y = self._select_dataset("test")
        pred, losses = self._get_predictions("test")

        if kind == "all":
            indices = [
                j[1]
                for j in sorted(
                    [
                        (x, i)
                        for (i, x) in enumerate(losses)
                        if x >= threshold
                    ],
                    reverse=True,
                )
            ]
            len_all = y.shape[0]
            print_statement = ""

        elif kind == "single":
            indices = [
                j[1]
                for j in sorted(
                    [
                        (x, i)
                        for (i, x) in enumerate(losses)
                        if (
                            len(np.where(y[i] == 0.0)[0]) == 3
                            and x >= threshold
                        )
                    ],
                    reverse=True,
                )
            ]
            len_all = len(
                [
                    i
                    for (i, x) in enumerate(losses)
                    if (len(np.where(y[i] == 0.0)[0]) == 3)
                ]
            )
            print_statement = "with a single species "

        elif kind == "linear_comb":
            indices = [
                j[1]
                for j in sorted(
                    [
                        (x, i)
                        for (i, x) in enumerate(losses)
                        if (
                            len(np.where(y[i] == 0.0)[0]) != 3
                            and x >= threshold
                        )
                    ],
                    reverse=True,
                )
            ]
            len_all = len(
                [
                    i
                    for (i, x) in enumerate(losses)
                    if (len(np.where(y[i] == 0.0)[0]) != 3)
                ]
            )
            print_statement = "with multiple species "

        if threshold > 0.0:
            print(
                "{0} of {1} test samples ({2}%) {3}have a mean ".format(
                    str(len(indices)),
                    str(len_all),
                    str(
                        100 * (np.around(len(indices) / len_all, decimals=3))
                    ),
                    print_statement,
                )
                + "absolute error of of at least {0}.".format(str(threshold))
            )
            indices = indices[-no_of_spectra:]

        else:
            indices = indices[:no_of_spectra]

        self.plot_spectra(
            no_of_spectra=no_of_spectra,
            dataset="test",
            indices=indices,
            with_prediction=True,
        )

    def show_wrong_classification(self):
        """
        Plot all spectra in the test data with a wrong prediction.

        Only works for classification.

        Returns
        -------
        None.

        """
        data = []
        texts = []

        wrong_pred_args = []

        for i in range(self.pred_test.shape[0]):
            argmax_class_true = np.argmax(self.y_test[i, :], axis=0)
            argmax_class_pred = np.argmax(self.pred_test[i, :], axis=0)

        if argmax_class_true != argmax_class_pred:
            wrong_pred_args.append(i)
        no_of_wrong_pred = len(wrong_pred_args)
        print(
            "No. of wrong predictions on the test data: "
            + str(no_of_wrong_pred)
        )

        if no_of_wrong_pred > 0:
            for i in range(no_of_wrong_pred):
                index = wrong_pred_args[i]

                real_y = "Real: " + str(self.y_test[index]) + "\n"
                # Round prediction and sum to 1
                tmp_array = np.around(self.pred_test[index], decimals=4)
                # row_sums = tmp_array.sum()
                # tmp_array = tmp_array / row_sums
                # tmp_array = np.around(tmp_array, decimals=2)
                pred_y = "Prediction: " + str(tmp_array) + "\n"
                pred_label = (
                    "Predicted label: "
                    + str(self.pred_test_classes[index, 0])
                    + "\n"
                )
                labels = self.y_test[index]
                for j, value in enumerate(labels):
                    if value == 1:
                        label = str(self.labels[j])
                        label = "Real label: " + label + "\n"
                text = real_y + pred_y + label + pred_label
                try:
                    sim = self._write_sim_text(dataset="test", index=index)
                    text += sim
                except AttributeError:
                    pass
                try:
                    name = self._write_measured_text(
                        dataset="test", index=index
                    )
                    text += name
                except AttributeError:
                    pass

                texts.append(text)

            data = np.array(data)

            graphic = SpectraPlot(data=data, annots=texts)
            fig, axs = graphic.plot()

    def plot_prob_predictions(
        self, prob_preds, dataset="test", no_of_spectra=10
    ):

        X, y = self._select_dataset(dataset_name="test")

        if no_of_spectra > y.shape[0]:
            print("Provide no. of spectra was bigger than dataset size.")
            no_of_spectra = y.shape[0]

        fig, axs = plt.subplots(
            nrows=no_of_spectra,
            ncols=5,
            figsize=(22, 5 * no_of_spectra),
        )

        max_y = np.max([np.float(np.max(y)), np.max(prob_preds)])

        random_numbers = []
        for i in range(no_of_spectra):
            ax0 = axs[i, 0]
            ax1 = axs[i, 1]
            ax2 = axs[i, 2]
            ax3 = axs[i, 3]
            ax4 = axs[i, 4]

            r = np.random.randint(0, X.shape[0])

            while r in random_numbers:
                r = np.random.randint(0, X.shape[0])
            random_numbers.append(r)

            if len(X.shape) == 4:
                ax0.imshow(X[r, :, :, 0], cmap="gist_gray")
            elif len(X.shape) == 3:
                ax0.plot(self.energies, self.X[r])
                ax0.invert_xaxis()
                ax0.set_xlim(np.max(self.energies), np.min(self.energies))
                ax0.set_xlabel("Binding energy (eV)")
                ax0.set_ylabel("Intensity (arb. units)")
                annot = self.write_text_for_spectrum(
                    dataset="test",
                    index=r,
                    with_prediction=False,
                )
                ax0.set_title("Spectrum no. {}".format(r))
                ax0.text(
                    0.025,
                    0.4,
                    annot,
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=ax0.transAxes,
                    fontsize=12,
                )

            sns.barplot(x=np.arange(self.num_classes), y=y[r], ax=ax1)
            # ax1.set_ylim([0, np.max(y)])
            ax1.set_title("Ground Truth")

            colors = iter(mcolors.CSS4_COLORS.keys())
            for pred in prob_preds[r, :20]:
                sns.barplot(
                    x=np.arange(self.num_classes),
                    y=pred,
                    color=next(colors),
                    alpha=0.2,
                    ax=ax2,
                )
            # ax2.set_ylim([0, max_y])
            ax2.set_title("Posterior Samples")

            for j, row in enumerate(prob_preds[r, :, :].transpose()):
                _ = ax3.hist(
                    row,
                    bins=25,
                    # range=(-2.,2.),
                    orientation="horizontal",
                    fill=True,
                    linewidth=1,
                    label=self.labels[j],
                )

            ax3.legend()
            ax3.set_xscale("log")
            ax3.set_xlabel("Prediction")
            ax3.set_ylabel("Counts")
            ax3.set_title("Prediction Histogram")

            ax4.bar(
                np.arange(self.num_classes),
                np.mean(prob_preds[r, :, :], axis=0),
                yerr=np.std(prob_preds[r, :, :], axis=0),
                align="center",
                ecolor="black",
                capsize=10,
            )

            ax4.set_title("Predictive Probabilities")

            for ax in [ax1, ax2, ax4]:
                ax.set_xticks(np.arange(self.num_classes))
                ax.set_xticklabels(self.labels)
        fig.tight_layout()

        return fig

    def _select_dataset(self, dataset_name):
        """
        Select a data set (for plotting).

        Parameters
        ----------
        name : str
            Name of the data set. Options: 'train', 'val', 'test'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if dataset_name == "train":
            X = self.X_train
            y = self.y_train

        elif dataset_name == "val":
            X = self.X_val
            y = self.y_val

        elif dataset_name == "test":
            X = self.X_test
            y = self.y_test

        return X, y

    def _get_predictions(self, dataset_name):
        """
        Get the predictions and losses for one data set.

        Used for writing the annotations during plotting.

        Parameters
        ----------
        name : str
            Name of the data set. Options: 'train', 'val', 'test'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if dataset_name == "train":
            pred = self.pred_train
            losses = self.losses_train

        elif dataset_name == "test":
            pred = self.pred_test
            losses = self.losses_test

        return pred, losses

    def write_text_for_spectrum(self, dataset, index, with_prediction=True):
        """
        Create the annotation for a plot of one spectrum.

        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.
        index : TYPE
            DESCRIPTION.
        with_prediction : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        text : TYPE
            DESCRIPTION.

        """
        X, y = self._select_dataset(dataset)

        label = str(np.around(y[index], decimals=3))
        text = "Real: " + label + "\n"

        if with_prediction:
            pred, losses = self._get_predictions(dataset)

            # Round prediction and sum to 1
            tmp_array = np.around(pred[index], decimals=4)
            # row_sums = tmp_array.sum()
            # tmp_array = tmp_array / row_sums
            # tmp_array = np.around(tmp_array, decimals=3)
            pred_text = "Prediction: " + str(list(tmp_array)) + "\n"
            text += pred_text

        try:
            text += self._write_sim_text(dataset=dataset, index=index)
        except AttributeError:
            pass
        try:
            text += self._write_measured_text(dataset=dataset, index=index)
        except AttributeError:
            pass

        if with_prediction:
            loss_text = (
                "\n" + "Loss: " + str(np.around(losses[index], decimals=3))
            )
            text += loss_text

        return text

    def _write_sim_text(self, dataset, index):
        """
        Generate a string containing the simulation parameters.

        Parameters
        ----------
        dataset : str
            Either 'train', 'val', or 'test.
            Needed for taking the correct sim_values.
        index : int
            Index of the example for which the text shall be created.

        Returns
        -------
        sim_text : str
            Output text in a figure.

        """
        if dataset == "train":
            shift_x = self.sim_values_train["shift_x"][index]
            noise = self.sim_values_train["noise"][index]
            fwhm = self.sim_values_train["fwhm"][index]

        elif dataset == "val":
            shift_x = self.sim_values_val["shift_x"][index]
            noise = self.sim_values_val["noise"][index]
            fwhm = self.sim_values_val["fwhm"][index]

        elif dataset == "test":
            shift_x = self.sim_values_test["shift_x"][index]
            noise = self.sim_values_test["noise"][index]
            fwhm = self.sim_values_test["fwhm"][index]

        if fwhm is not None and fwhm != 0:
            fwhm_text = (
                "FHWM: " + str(np.round(float(fwhm), decimals=2)) + ", "
            )
        else:
            fwhm_text = "FHWM: not changed" + ", "

        if shift_x is not None and shift_x != 0:
            shift_text = " Shift: " + "{:.2f}".format(float(shift_x)) + ", "
        else:
            shift_text = " Shift: none" + ", "

        if noise is not None and noise != 0:
            noise_text = "S/N: " + "{:.1f}".format(float(noise))
        else:
            noise_text = "S/N: not changed"

        sim_text = fwhm_text + shift_text + noise_text + "\n"

        if "scatterer" in self.sim_values.keys():
            sim_text += self._write_scatter_text(dataset, index)
        else:
            sim_text += "Scattering: none."

        return sim_text

    def _write_scatter_text(self, dataset, index):
        """
        Generate a string containing the scattering parameters.

        Parameters
        ----------
        dataset : str
            Either 'train', 'val', or 'test.
            Needed for taking the correct sim_values.
        index : int
            Index of the example for which the text shall be created.

        Returns
        -------
        scatter_text : str
            Output text in a figure.

        """
        if dataset == "train":
            scatterer = self.sim_values_train["scatterer"][index]
            distance = self.sim_values_train["distance"][index]
            pressure = self.sim_values_train["pressure"][index]

        elif dataset == "val":
            scatterer = self.sim_values_val["scatterer"][index]
            distance = self.sim_values_val["distance"][index]
            pressure = self.sim_values_val["pressure"][index]

        elif dataset == "test":
            scatterer = self.sim_values_test["scatterer"][index]
            distance = self.sim_values_test["distance"][index]
            pressure = self.sim_values_test["pressure"][index]

        scatterers = {"0": "He", "1": "H2", "2": "N2", "3": "O2"}
        try:
            scatterer_name = scatterers[str(scatterer[0])]
        except KeyError:
            return "Scattering: none."

        name_text = "Scatterer: " + scatterer_name + ", "

        pressure_text = "{:.1f}".format(float(pressure)) + " mbar, "

        distance_text = "d = " + "{:.1f}".format(float(distance)) + " mm"

        scatter_text = name_text + pressure_text + distance_text + "\n"

        return scatter_text

    def _write_measured_text(self, dataset, index):
        """
        Generate information about the measured spectra.

        Parameters
        ----------
        dataset : str
            Either 'train', 'val', or 'test.
            Needed for taking the correct names.
        index : int
            Index of the example for which the text shall be created.

        Returns
        -------
        measured_text : str
            Output text in a figure.

        """
        if dataset == "train":
            name = self.names_train[index][0]

        elif dataset == "val":
            name = self.names_val[index][0]

        elif dataset == "test":
            name = self.names_test[index][0]

        measured_text = "Spectrum no. " + str(index) + "\n" + str(name) + "\n"

        return measured_text


#%%
if __name__ == "__main__":
    np.random.seed(502)
    input_filepath = r"C:\Users\pielsticker\Simulations\20210520_Fe_linear_combination_small_gas_phase\20210520_Fe_linear_combination_small_gas_phase.h5"

    datahandler = DataHandler(intensity_only=True)
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
    ) = datahandler.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )
    print("Input shape: " + str(datahandler.input_shape))
    print("Labels: " + str(datahandler.labels))
    print("No. of classes: " + str(datahandler.num_classes))

    datahandler.plot_random(
        no_of_spectra=20, dataset="train", with_prediction=False
    )
