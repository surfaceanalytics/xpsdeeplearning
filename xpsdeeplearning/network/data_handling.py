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
Data handler for Classifier class.
"""
import numpy as np
import h5py
import sklearn

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from xpsdeeplearning.network.utils import ClassDistribution, SpectraPlot


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
        select_random_subset=True,
        shuffle=True,
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
        select_random_subset: bool
            Whether or not a random subset of the data shall
            be loaded.
            Default is True.
        shuffle : bool
            Whether or not the data should be shuffled randomly.
            Default is True.

        Returns
        -------
        return_data: tuple
            Tuple with train, val, test sets for each key in the
            HDF5 file.
            Always contains:
                X_train, X_val, y_test: arrays. Shape: 3d Numpy arrays.
                Features (spectra).
                y_train, y_val, y_test: arrays. Shape: 2d Numpy array.
                Labels (quantification).
            Depending on the type of file (measured/simulated),
            the method can also return more information about
            the data set:
                self.sim_values_train,
                self.sim_values_val,
                self.sim_values_test : dicts
                    Dictionaries containing information about the parameters
                    used during the artificical construction of the dataset.
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
        self.shuffle = shuffle
        self.select_random_subset = select_random_subset

        loaded_data = self._load_data_from_file()

        if self.shuffle:
            loaded_data = sklearn.utils.shuffle(
                *loaded_data, random_state=np.random.RandomState(seed=1)
            )

        # Store shuffled data (if shuffle=True), then split in train, val, test.
        self.X = loaded_data[0]
        self.X_train, self.X_val, self.X_test = self._split_test_val_train(self.X)

        self.y = loaded_data[1]

        self.y_train, self.y_val, self.y_test = self._split_test_val_train(self.y)

        return_data = [
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ]

        # Determine the shape of the training features,
        # needed for model building in Keras.
        self.input_shape = self.X_train.shape[1:]

        if self.sim_values:
            self.sim_values_train = {}
            self.sim_values_val = {}
            self.sim_values_test = {}

            # Store shuffled data
            for i, key in enumerate(self.sim_values.keys()):
                self.sim_values[key] = loaded_data[i + 2]

            # Split in train, val, test.
            for key, array in self.sim_values.items():
                (
                    array_train,
                    array_val,
                    array_test,
                ) = self._split_test_val_train(array)
                self.sim_values_train[key] = array_train
                self.sim_values_val[key] = array_val
                self.sim_values_test[key] = array_test
            return_data.extend(
                [
                    self.sim_values_train,
                    self.sim_values_val,
                    self.sim_values_test,
                ]
            )

        if len(self.names):
            self.names = loaded_data[-1]
            (
                self.names_train,
                self.names_val,
                self.names_test,
            ) = self._split_test_val_train(self.names)
            return_data.extend([self.names_train, self.names_val, self.names_test])

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

        return return_data

    def _load_data_from_file(self):
        """
        Load the data from the HDF5 file and preprocess it.

        Raises
        ------
        ValueError
            If the selected number of examples is bigger than the
            size of the X array in the HDF5 file.

        Returns
        -------
        loaded_data : list
            List of data that were in the HDF5 file.

        """
        with h5py.File(self.input_filepath, "r") as hf:
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
                    self.labels = [label.decode("utf-8") for label in hf["labels"][:]]
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
                if self.select_random_subset:
                    r = np.random.randint(0, dataset_size - self.no_of_examples + 1)
                else:
                    r = 0

            except ValueError as e:
                error_msg = (
                    "There are not enough spectra in this data set. "
                    + f"Please choose a value of no more than {dataset_size} "
                    + "for no_of_examples."
                )
                raise type(e)(error_msg)

            X = hf["X"][r : r + self.no_of_examples]
            X = X.astype(float)
            y = hf["y"][r : r + self.no_of_examples]

            if not self.intensity_only:
                new_energies = np.tile(
                    np.reshape(np.array(self.energies), (-1, 1)),
                    (X.shape[0], 1, 1),
                )
                X = np.dstack((new_energies, X))

            self.X = X
            self.y = y
            self.sim_values = {}
            self.names = []

            loaded_data = [X, y]

            # Check if the data set was artificially created.
            if "shiftx" in hf.keys():
                shift_x = hf["shiftx"][r : r + self.no_of_examples]
                noise = hf["noise"][r : r + self.no_of_examples]
                fwhm = hf["FWHM"][r : r + self.no_of_examples]

                loaded_data.extend([shift_x, noise, fwhm])

                self.sim_values["shift_x"] = shift_x
                self.sim_values["noise"] = noise
                self.sim_values["fwhm"] = fwhm

                # If the data set was artificially created, check
                # if scattering in a gas phase was simulated.
                if "scatterer" in hf.keys():
                    scatterer = hf["scatterer"][r : r + self.no_of_examples]
                    distance = hf["distance"][r : r + self.no_of_examples]
                    pressure = hf["pressure"][r : r + self.no_of_examples]

                    loaded_data.extend([scatterer, distance, pressure])

                    self.sim_values["scatterer"] = scatterer
                    self.sim_values["distance"] = distance
                    self.sim_values["pressure"] = pressure

            # Check if the spectra have associated names. Typical for
            # measured spectra.
            elif "names" in hf.keys():
                names_load_list = [
                    name[0].decode("utf-8")
                    for name in hf["names"][r : r + self.no_of_examples, :]
                ]
                self.names = np.reshape(np.array(names_load_list), (-1, 1))
                loaded_data.extend([self.names])

        return loaded_data

    def _split_test_val_train(self, data_array):
        """
        Split a numpy array into train, val, and test sets.

        First, the whole data is split into the train+val and test sets
        according to the attribute self.train_test_split. Secondly.
        the train+val sets are further split into train and val sets
        according to the attribute self.train_val_split.

        Parameters
        ----------
        data_array : ndarray
            N-dimensional array.

        Returns
        -------
        (d_train, d_val, d_test): tuple
            Three output arras are returned.
            E.g. for input X, the returns are X_train, X_val, X_test.
         : TYPE
            DESCRIPTION.
        d_test : TYPE
            DESCRIPTION.

        """
        # First split into train+val and test sets
        no_of_train_val = int((1 - self.train_test_split) * data_array.shape[0])
        d_train_val = data_array[:no_of_train_val]
        d_test = data_array[no_of_train_val:]

        # Then create val subset from train set
        no_of_train = int((1 - self.train_val_split) * d_train_val.shape[0])
        d_train = d_train_val[:no_of_train]
        d_val = d_train_val[no_of_train:]

        return (d_train, d_val, d_test)

    def _only_keep_classification_data(self):
        """
        Keep only data with just one species, i.e. with one label of 1.0
        and the rest of the labels = 0.0.
        """
        indices = np.where(self.y == 1.0)[0]

        print(indices)

        self.X, self.y = self.X[indices], self.y[indices]

        self.X_train, self.X_val, self.X_test = self._split_test_val_train(self.X)
        self.y_train, self.y_val, self.y_test = self._split_test_val_train(self.X)

        return_data = [
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ]

        if self.sim_values:
            new_sim_values = {}

            for key, sim_arrays in self.sim_values.items():
                new_sim_values[key] = sim_arrays[indices]

            self.sim_values = new_sim_values

            self.sim_values_train = {}
            self.sim_values_val = {}
            self.sim_values_test = {}

            # Split in train, val, test.
            for key, array in self.sim_values.items():
                (
                    array_train,
                    array_val,
                    array_test,
                ) = self._split_test_val_train(array)

                self.sim_values_train[key] = array_train
                self.sim_values_val[key] = array_val
                self.sim_values_test[key] = array_test

            return_data.extend(
                [
                    self.sim_values_train,
                    self.sim_values_val,
                    self.sim_values_test,
                ]
            )

        if self.names:
            self.names = self.names[indices]

            (
                self.names_train,
                self.names_val,
                self.names_test,
            ) = self._split_test_val_train(self.names)
            return_data.extend([self.names_train, self.names_val, self.names_test])

        print(
            "Only spectra with one species were left in the data set! Test/val/train splits were kept."
        )
        print(f"Remaining no. of training examples: {self.y_train.shape[0]}")
        print(f"Remaining no. of val examples: {self.y_val.shape[0]}")
        print(f"Remaining no. of test examples: {self.y_test.shape[0]}")

        return return_data

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
            label = str(np.around(y[index], decimals=3))
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

    def show_worst_predictions(self, no_of_spectra, kind="all", threshold=0.0):
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
                    [(x, i) for (i, x) in enumerate(losses) if x >= threshold],
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
                        if (len(np.where(y[i] == 0.0)[0]) == 3 and x >= threshold)
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
                        if (len(np.where(y[i] == 0.0)[0]) != 3 and x >= threshold)
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
                    str(100 * (np.around(len(indices) / len_all, decimals=3))),
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
        print("No. of wrong predictions on the test data: " + str(no_of_wrong_pred))

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
                    "Predicted label: " + str(self.pred_test_classes[index, 0]) + "\n"
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
                    name = self._write_measured_text(dataset="test", index=index)
                    text += name
                except AttributeError:
                    pass

                texts.append(text)

            data = np.array(data)

            graphic = SpectraPlot(data=data, annots=texts)
            fig, axs = graphic.plot()

    def plot_prob_predictions(
        self, prob_preds, indices, dataset="test", no_of_spectra=10
    ):
        X, y = self._select_dataset(dataset_name="test")

        if no_of_spectra > y.shape[0]:
            print("Provided no. of spectra was bigger than dataset size.")
            no_of_spectra = y.shape[0]

        fig, axs = plt.subplots(
            nrows=no_of_spectra,
            ncols=5,
            figsize=(22, 5 * no_of_spectra),
        )

        for i in range(no_of_spectra):
            ax0 = axs[i, 0]
            ax1 = axs[i, 1]
            ax2 = axs[i, 2]
            ax3 = axs[i, 3]
            ax4 = axs[i, 4]

            index = indices[i]

            if len(X.shape) == 4:
                ax0.imshow(X[index, :, :, 0], cmap="gist_gray")
            elif len(X.shape) == 3:
                ax0.plot(self.energies, self.X[index])
                ax0.invert_xaxis()
                ax0.set_xlim(np.max(self.energies), np.min(self.energies))
                ax0.set_xlabel("Binding energy (eV)")
                ax0.set_ylabel("Intensity (arb. units)")
                annot = self.write_text_for_spectrum(
                    dataset="test",
                    index=index,
                    with_prediction=False,
                )
                ax0.set_title("Spectrum no. {}".format(index))
                ax0.text(
                    0.025,
                    0.4,
                    annot,
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=ax0.transAxes,
                    fontsize=12,
                )

            sns.barplot(x=np.arange(self.num_classes), y=y[index], ax=ax1)
            ax1.set_title("Ground Truth")

            colors = iter(mcolors.CSS4_COLORS.keys())
            for pred in prob_preds[index, :20]:
                sns.barplot(
                    x=np.arange(self.num_classes),
                    y=pred,
                    color=next(colors),
                    alpha=0.2,
                    ax=ax2,
                )
            # ax2.set_ylim([0, max_y])
            ax2.set_title("Posterior Samples")

            for j, row in enumerate(prob_preds[index, :, :].transpose()):
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
                np.mean(prob_preds[index, :, :], axis=0),
                yerr=np.std(prob_preds[index, :, :], axis=0),
                align="center",
                ecolor="black",
                capsize=10,
            )

            ax4.set_title("Predictive Probabilities")

            for ax in [ax1, ax2, ax4]:
                ax.set_xticks(np.arange(self.num_classes))
                ax.set_xticklabels(self.labels)
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_ylim(0, 1.05)

        fig.tight_layout()

        return fig

    def _select_prob_predictions(
        self, prob_preds, kind, dataset="test", no_of_spectra=10
    ):
        if kind == "random":
            X, y = self._select_dataset(dataset_name="test")
            indices = []
            for i in range(no_of_spectra):
                r = np.random.randint(0, X.shape[0])
                while r in indices:
                    r = np.random.randint(0, X.shape[0])
                indices.append(r)
            return indices

        elif kind == "min":
            return [
                j[1]
                for j in sorted(
                    [(p, i) for (i, p) in enumerate(np.std(prob_preds, axis=1)[:, 0])],
                    reverse=False,
                )
            ]

        elif kind == "max":
            return [
                j[1]
                for j in sorted(
                    [(p, i) for (i, p) in enumerate(np.std(prob_preds, axis=1)[:, 0])],
                    reverse=True,
                )
            ]

        else:
            print("Select a valid kind!")

    def _select_dataset(self, dataset_name):
        """
        Select a data set (for plotting).

        Parameters
        ----------
        name : str
            Name of the data set. Options: 'train', 'val', 'test'.

        Returns
        -------
        X,y
           Features (spectra) and quantification labels of that data
           set.

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
        pred, losses
            Predictions and losses for one data set..

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
        dataset : str
            Name of the data set. Options: 'train', 'val', 'test'..
        index : int
            Index of the spectrum.
        with_prediction : bol, optional
            If true, the predicted quantification is also written.
            The default is True.

        Returns
        -------
        text : str
            Descriptive text for one spectrum.

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
            loss_text = "\n" + "Loss: " + str(np.around(losses[index], decimals=3))
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
            fwhm_text = "FHWM: " + str(np.round(float(fwhm), decimals=2)) + ", "
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
