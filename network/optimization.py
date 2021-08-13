# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 12:37:52 2020.

@author: pielsticker
"""
import numpy as np
import talos
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns

import xpsdeeplearning.network.classifier as classifier

#%%
class Hyperoptimization:
    """Optimization of the Keras hyperparameters of a using Talos."""

    def __init__(self, time, exp_name):
        """
        Initialize empty scans list and DataFrame.

        Parameters
        ----------
        time : str
            Time of experiment start.
        exp_name : str
            Name of the experimental run.

        Returns
        -------
        None.

        """
        self.time = time
        self.exp_name = exp_name

        root_dir = os.getcwd()
        self.dir_name = self.time + "_" + self.exp_name
        self.test_dir = os.path.join(
            *[root_dir, "runs", self.dir_name, "param_tests"]
        )

        self.scans = []
        self.full_data = pd.DataFrame()

        if os.path.isdir(self.test_dir) is False:
            os.makedirs(self.test_dir)
            if os.path.isdir(self.test_dir):
                print(
                    "Test folder created at "
                    + str(self.test_dir.split(root_dir)[1])
                )
        else:
            print(
                "Test folder was already at "
                + str(self.test_dir.split(root_dir)[1])
            )

        self.save_full_data()

    def initialize_clf(self, task="regression", intensity_only=True):
        """
        Initialize a Classifier object.

        Parameters
        ----------
        label_values : list
            List of strings of the labels of the XPS spectra.
            Example: 
                label_values = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']

        Returns
        -------
        None.

        """
        self.clf = classifier.Classifier(
            time=self.time,
            exp_name=self.exp_name,
            task=task,
            intensity_only=intensity_only,
        )

        self.fig_dir = self.clf.logging.fig_dir

    def hyper_opt(self, x_train, y_train, x_val, y_val, params):
        """
        Instantiate, compile and call model.
        
        Used in the Talos Scan.

        Parameters
        ----------
        x_train : sequence of array_like
            3d array, or a list of arrays with features for the
            prediction task.
        y_train : ndarray
            2d array, or a list of arrays with labels for the 
            prediction task.
        x_val : sequence of array_like
            3d array, or a list of arrays with features for the 
            validation.
        y_val : ndarray
            2d array, or a list of arrays with labels for the
            validation.
        params : dict
           Dictionary containing all parameters that should be tested 
           in the Talos Scan, a subset of which will be selected at 
           random for training and evaluation.
           For example:
            p = {'lr': (0.5, 5, 10),
                 'first_neuron':[4, 8, 16, 32, 64],
                 'hidden_layers':[0, 1, 2]}.
           Need to know the model architecture beforehand.

        Returns
        -------
        out : tf.keras.callbacks.History 
            Training history object from Keras.
        model : tensorflow.keras.models.Model
            Keras model.

        """
        model_class = type(self.clf.model)
        model = model_class(
            self.clf.datahandler.input_shape,
            self.clf.datahandler.num_classes,
            params,
        )
        model.compile(
            loss=params["loss_function"](),
            optimizer=params["optimizer"](params["learning_rate"]),
        )

        out = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
        )

        return out, model

    def scan_parameter_space(self, params, **kwargs):
        """
        Scan the parameter space provided in the params dictionary.
        
        Keywords can be used to limit the search space.

        Parameters
        ----------
        params : dict
            Parameter space to be scanned.
        **kwargs : str
            Limiter arguments in Talos (see Talos doc), e.g.
            'fraction_limit', 'round_limit', 'time_limit' 

        Returns
        -------
        None.

        """
        # Restore all previous scans to the scan list.
        self.scans = self.restore_previous_scans()

        # Calculate the number of the scan according to the files
        # already in the test directory.
        filenames = next(os.walk(self.test_dir))[2]
        number = 0
        for filename in filenames:
            if filename.startswith("test_log") and filename.endswith(".pkl"):
                number += 1

        # Initialize Scan object (based on talos.Scan)
        # start runtime

        scan = Scan(
            x=self.clf.datahandler.X_train,
            y=self.clf.datahandler.y_train,
            params=params,
            model=self.hyper_opt,
            experiment_name=self.dir_name,
            x_val=self.clf.datahandler.X_val,
            y_val=self.clf.datahandler.y_val,
            val_split=0,
            reduction_metric="val_loss",
            minimize_loss=True,
            number=number,
            **kwargs
        )
        self.scans.append(scan)

        try:
            from talos.scan.scan_run import scan_run

            scan_run(scan)
            print(
                (
                    "\n Parameter space was scanned. "
                    + "Training log was saved."
                )
            )

        except KeyboardInterrupt:
            # In case of interruption, still finish the scan round.
            scan.interrupt()
            print(
                (
                    "\n Scan was interrupted! "
                    + "Training log was saved until round {0}."
                ).format(scan.data.shape[0])
            )

        scan.deploy()

        # Combine the new data with the previous results.
        self.full_data = self.combine_results()
        # Save the full data to pickle and csv files.
        self.save_full_data()
        # Store the saved_weights from all scans.
        self.all_weights = self._get_all_weights()

    def restore_previous_scans(self):
        """
        Create a list of RestoredScan objects.
        
        This is used to restore information about previous results 
        on this data set.

        Returns
        -------
        scans : list
            Restored Scans.

        """
        scans = []
        filenames = [
            filename
            for filename in next(os.walk(self.test_dir))[2]
            if (filename.startswith("test_log") and filename.endswith("pkl"))
        ]
        
        for filename in filenames:
            scan_number = int(filename.split(".")[0][-1])
            restored_scan = RestoredScan(self.test_dir, scan_number)
            scans.append(restored_scan)
        
        if not scans:
            print("There were no previous scans.")

        return scans

    def combine_results(self):
        """
        Combine the results from all scans in the self.scans list.
        
        Return a DataFrame containing all results.

        Returns
        -------
        full_data : pandas.DataFrame
            A Dataframe containing the results from all scan objects.

        """
        filenames = next(os.walk(self.test_dir))[2]
        full_data = pd.DataFrame()
        for filename in filenames:
            if filename.startswith("test_log") and filename.endswith("pkl"):
                results_file = os.path.join(self.test_dir, filename)
                new_df = pd.read_pickle(results_file)
                full_data = pd.concat([full_data, new_df], ignore_index=True)

        return full_data

    def save_full_data(self):
        """
        Save the full data to CSV and pickle files.

        Returns
        -------
        None.

        """
        new_csv_filepath = os.path.join(self.test_dir, "all_tests.csv")
        new_pkl_filepath = os.path.join(self.test_dir, "all_tests.pkl")

        np.savetxt(new_csv_filepath, self.full_data, fmt="%s", delimiter=",")

        self.full_data.to_pickle(new_pkl_filepath)

    def _get_all_weights(self):
        """
        Combine the weights from all scans into one numpy array.

        Returns
        -------
        array
            Concatenated weights of all scans.

        """
        weights = [scan.saved_weights for scan in self.scans]

        return np.concatenate(weights)

    def initialize_analyzer(self):
        """
        Initialize an analyzer object using the full_data DataFrame.

        Returns
        -------
        None.

        """
        try:
            df = self.full_data.drop(
                ["start", "end", "duration"], axis=1, inplace=False
            )
        except KeyError:
            df = self.full_data

        self.analyzer = Analysis(df)

    def get_best_params(self, metric):
        """
        Get the best params for a certain metric.
        
        Exploits the Analyzer object.

        Parameters
        ----------
        metric : str
            Metric to evaluate. Typically 'val_acc' or 'val_loss'.

        Returns
        -------
        dict
            Dictionary containing the best parameters across all scans.

        """
        best_round = self.analyzer._get_best_round(metric)
        self.best_params = self.analyzer._get_params_from_id(best_round)

        return self.best_params

    def load_model_from_scans(
        self, best=False, model_id=None, metric="val_loss"
    ):
        """
        Reload a model from all scans.
        
        Either the best model or a model from an ID is loaded.

        Parameters
        ----------
        best : bool, optional
            If True, load the best model (by the given metric).
            The default is False.
        model_id : int, optional
            ID of the model. Only if best == False.
            The default is None.
        metric : str, optional
            If best, the metric for which to choose the best model.
            The default is 'val_loss'.

        Returns
        -------
        None.

        """
        if best is True or model_id is None:
            model_id = self.analyzer._get_best_round(metric)

        params = self.analyzer._get_params_from_id(model_id)
        model_class = type(self.clf.model)
        self.clf.model = model_class(
            self.clf.datahandler.input_shape,
            self.clf.datahandler.num_classes,
            params,
        )
        self.clf.model.compile(
            loss=params["loss_function"](),
            optimizer=params["optimizer"](params["learning_rate"]),
        )

        self.clf.model._name = "Model_{0}".format(model_id)
        self.clf.model.set_weights(self.all_weights[model_id])


class Scan(talos.Scan):
    """Class for performing a scan of the parameter space."""

    def __init__(
        self,
        x,
        y,
        params,
        model,
        experiment_name,
        x_val=None,
        y_val=None,
        val_split=0.3,
        random_method="uniform_mersenne",
        seed=None,
        performance_target=None,
        fraction_limit=None,
        round_limit=None,
        time_limit=None,
        boolean_limit=None,
        reduction_method=None,
        reduction_interval=50,
        reduction_window=20,
        reduction_threshold=0.2,
        reduction_metric="val_loss",
        minimize_loss=True,
        disable_progress_bar=False,
        print_params=True,
        clear_session=True,
        save_weights=True,
        number=0,
    ):
        """
        Overwrite __init__ method taken from talos.Scan.
        
        In comparison to original talos.Scan, scan_round is
        not run on implementation (see talos docs),
        """
        super(Scan, self).__init__(
            x=x,
            y=y,
            params=params,
            model=model,
            experiment_name=experiment_name,
            x_val=x_val,
            y_val=y_val,
            val_split=val_split,
            random_method=random_method,
            seed=seed,
            performance_target=performance_target,
            fraction_limit=fraction_limit,
            round_limit=round_limit,
            time_limit=time_limit,
            boolean_limit=boolean_limit,
            reduction_method=reduction_method,
            reduction_interval=reduction_interval,
            reduction_window=reduction_window,
            reduction_threshold=reduction_threshold,
            reduction_metric=reduction_metric,
            minimize_loss=minimize_loss,
            disable_progress_bar=disable_progress_bar,
            print_params=print_params,
            clear_session=clear_session,
            save_weights=save_weights,
            number=number,
        )

    def deploy(self):
        """
        Save all relevant information from the Scan object.

        Method very similar to the Deploy class of Talos.
        Saves all details, the model, information about the rounds 
        and the weights to different files. Needed for restoring of
        scans.

        Parameters
        ----------
        test_dir : str
            Directory where the scan is stored.

        Returns
        -------
        None.

        """
        scan_folder = "scan" + str(self.number)
        scan_dir = os.path.join(self.experiment_dir, scan_folder)
        if os.path.isdir(scan_dir) is False:
            os.makedirs(scan_dir)

        detail_file = os.path.join(scan_dir, "details.txt")
        param_file = os.path.join(scan_dir, "params")
        weights_file = os.path.join(scan_dir, "saved_weights")
        round_file = os.path.join(scan_dir, "round_log.csv")
        model_file = os.path.join(scan_dir, "saved_models.txt")

        # Save information about the details of the scan.
        self.details.to_csv(detail_file)
        # Save all parameters.
        np.save(param_file, self.params)
        # Save the weights of each round in array format.
        np.save(weights_file, self.saved_weights)

        # Save information about the round history.
        with open(round_file, "w") as roundfile:
            round_data = pd.DataFrame()
            round_data["loss"] = self.learning_entropy.loss
            round_data["start"] = self.round_times.start
            round_data["end"] = self.round_times.end
            round_data["duration"] = self.round_times.duration
            round_data["round_history"] = self.round_history
            round_data.to_csv(roundfile)

        # Save a list of the model parameters used in each round.
        with open(model_file, "w") as modelfile:
            for listitem in self.saved_models:
                modelfile.write("%s\n" % listitem)

        print("Data was saved to scan archive.")

    def interrupt(self):
        """Catch an interruption and still save the data."""
        self.round_params = self.param_object.round_parameters()

        from talos.scan.scan_round import scan_round

        scan_round(self)

        self.pbar.close()

        from talos.logging.logging_finish import logging_finish

        logging_finish(self)
        from talos.scan.scan_finish import scan_finish

        scan_finish(self)


class RestoredScan:
    """Class for restoring information about previous scans."""

    def __init__(self, folder, number):
        """
        Load all data from a previous scan.
        
        This class is to be used if the hyperparamter scan had to 
        be abandoned, but shall be continued in a new Scan object.
        
        Note: It is not possible (yet) to run the scan in this class,
        this class is only a container for the results from a previous 
        scan.

        Parameters
        ----------
        folder : str
            Directory of the stored scans.
        number : int
            Number of the scan to load. Will search the filenames in
            the folder directory for this number.

        Returns
        -------
        None.

        """
        self.number = number
        scan_folder = "scan" + str(self.number)
        scan_dir = os.path.join(folder, scan_folder)
        detail_file = os.path.join(scan_dir, "details.txt")
        weights_file = os.path.join(scan_dir, "saved_weights.npy")
        round_file = os.path.join(scan_dir, "round_log.csv")
        model_file = os.path.join(scan_dir, "saved_models.txt")

        self.details = pd.read_csv(
            detail_file, header=None, squeeze=True, index_col=0
        )

        self.saved_weights = np.load(weights_file, allow_pickle=True)

        round_data = pd.read_csv(round_file)
        self.learning_entropy = pd.DataFrame(round_data["loss"])

        self.round_times = round_data[["start", "end", "duration"]]

        self.round_history = list(round_data["round_history"])

        self.saved_models = []
        with open(model_file, "r") as modelfile:
            for line in modelfile.readlines():
                self.saved_models.append(line[:-1])

        print("Previous data for Scan {} was loaded!".format(self.number))


class Analysis:
    """Class for the analysis of results from one or multiple scans."""

    def __init__(self, df):
        """
        Initialize the attribute self.df from an results DataFrame.
        
        This class is very similar to talos.Reporting, but with added 
        analysis tools.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the results of one or multiple scans.

        Returns
        -------
        None.

        """
        self.df = df

    def _get_params_from_id(self, model_id):
        """
        Load the parameters for a particular round.

        Parameters
        ----------
        model_id : int
            Row to choose the parameters.

        Returns
        -------
        params : dict
            Dictionary of all parameters used in the rounds..

        """
        params = dict(self.df.iloc[model_id])

        return params

    def _get_best_round(self, metric, low=True):
        """
        Return the ID of the round with the best value for a metric.

        Parameters
        ----------
        metric : str
            Metric to evaluate. Typically 'val_acc' or 'val_loss'.
        low : bool, optional
            If low, the minimum value of the metric in the results df
            is searched.
            If low == False, the maximum value is searched.
            The default is True.

        Returns
        -------
        int
            ID of the best round.

        """
        return self.df[self.df[metric] == self.df[metric].min()].index[0]

    def _minimum_value(self, metric):
        """Return the minimum value for a given metric."""
        return min(self.df[metric])

    def _exclude_unchanged(self):
        """Exlude the parameters not changed during the scans."""
        exclude = []
        for key, values in self.df.iteritems():
            if values.nunique() == 1:
                exclude.append(key)

        return exclude

    def _cols(self, metric, exclude):
        """Remove other than desired metric from data table."""
        cols = [
            col for col in self.df.columns if col not in exclude + [metric]
        ]

        # make sure only unique values in col list
        cols = list(set(cols))

        return cols

    def create_line_data(self, metric):
        """
        Select the metric column of the df for a line plot.

        Parameters
        ----------
        metric : str
            Metric to evaluate. Typically 'val_acc' or 'val_loss'.

        Returns
        -------
        df
            Data of the metric across all rounds.

        """
        return self.df[metric]

    def correlate(self, metric):
        """
        Return a correlation matrix between parameters and a metric.

        Parameters
        ----------
        metric : str
            Metric to evaluate. Typically 'val_acc' or 'val_loss'.

        Returns
        -------
        corr_data : pandas.DataFrame
            DataFrame with pairwise correlation of columns.

        """
        exclude = self._exclude_unchanged()
        out = self.df[self._cols(metric, exclude)]

        out.insert(0, metric, self.df[metric])

        corr_data = out.corr(method="pearson")

        return corr_data

    def create_hist_data(self, metric):
        """
        Select the metric column of the df for a histogram plot.

        Parameters
        ----------
        metric : str
            Metric to evaluate. Typically 'val_acc' or 'val_loss'.

        Returns
        -------
        df
            Data of the metric across all rounds.

        """
        return self.df[metric]

    def create_kde_data(self, x, y=None):
        """
        Select one or two columns of the df for a KDE plot.

        Parameters
        ----------
        x : str
            Column for a 1d or 2d KDE plot.
        y : str, optional
            Second column for a 1d or 2d KDE plot.
            If None, the plot is only 1d.
            The default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame only containing one or two columns of the main df.
        """
        if y is not None:
            return self.df[[x, y]]
        return self.df

    def create_bar_data(self, x, y, hue, col):
        """
        Select four columns of the df for a 4d bar plot.

        Parameters
        ----------
        x : str
            Column to be used on the x-axis of the plot.
        y : str
            Column to be used on the y-axis of the plot.
        hue : str
            Column to be used as grouping variable in the plot.
        col : str
            Column to be used for coloration of the plot.

        Returns
        -------
        d.DataFrame
            DataFrame only containing four columns of the main df.
        """
        return self.df[[x, y, hue, col]]


class Plot:
    """Base class for plotting one of the analysis plots."""

    def __init__(self, data):
        """
        Initialize empty data and figure.

        Fonts and display names are set here.

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.data = data
        self.name = None
        self.fig = None

        self.font_dict = {
            "fontsize": 15,
            "fontweight": 1,
            "color": "black",
            "verticalalignment": "bottom",
            "horizontalalignment": "center",
        }

        display_names = {
            "val_acc": "Validation accuracy",
            "val_loss": "Validation loss",
            "loss": "Loss",
        }

        try:
            self.metric = self.data.name
            self.metric_display_name = display_names[self.metric]
        except AttributeError:
            pass

    def to_file(self, filepath):
        """
        Save the figure stored in self.fig to a PNG file.

        Parameters
        ----------
        filepath : str
            Filepath for saving the figure.

        Returns
        -------
        None.

        """
        filename = self.name + ".png"
        self.fig_dir = os.path.join(filepath, filename)
        self.fig.savefig(self.fig_dir)
        print("File saved to figures directory!")


class LinePlot(Plot):
    """Class for a line plot of a chosen metric across all rounds."""

    def __init__(self, data):
        """
        Initialize data and name attributes.

        Parameters
        ----------
        data : pd.DataFrame
            Subset of a dataframe containing the relevant data for a 
            line plot.

        Returns
        -------
        None.

        """
        super(LinePlot, self).__init__(data)
        self.data = data
        self.name = "line plot_" + self.metric

    def plot(self):
        """
        Create a line plot of the data.
        
        Returns
        -------
        None.

        """
        self.x = range(self.data.shape[0])

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        self.ax.plot(
            self.x,
            self.data,
            drawstyle="default",
            linestyle="solid",
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgewidth=1,
            mfc="blue",
            rasterized=True,
            aa=True,
            alpha=1,
        )

        self.ax.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins=5, integer=True)
        )

        self.ax.set_xlabel("Round", self.font_dict)

        self.ax.set_title(
            self.metric_display_name + " across all rounds", self.font_dict
        )
        self.ax.set_ylabel(self.metric_display_name, self.font_dict)


class HistPlot(Plot):
    """Class for a histogram of a chosen metric across all rounds."""

    def __init__(self, data):
        """
        Initialize the super-class Plot and the attribute name.

        Parameters
        ----------
        data : pd.DataFrame
            Subset of a dataframe containing the relevant data for a 
            histogram.

        Returns
        -------
        None.

        """
        super(HistPlot, self).__init__(data)
        self.name = "histogram_" + self.metric

    def plot(self, bins=10):
        """
        Plot a histogram of the data across all rounds.

        Parameters
        ----------
        bins : int, optional
            Number of bins in the histogram.
            The default is 10.

        Returns
        -------
        None.

        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.hist(self.data, bins=bins)

        self.ax.set_title(
            "Histogram of the {0} across all rounds".format(
                self.metric_display_name
            ),
            self.font_dict,
        )

        self.ax.set_xlabel(
            self.metric_display_name,
            fontdict={"fontsize": 15, "fontweight": 1, "color": "black"},
        )
        self.ax.set_ylabel("Counts", self.font_dict)

        self.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))


class CorrPlot(Plot):
    """Correlation plot between all parameters and a metric."""

    def __init__(self, data):
        """
        Initialize the super-class Plot and the attribute name.

        Parameters
        ----------
        data : pd.DataFrame
            Subset of a dataframe containing the relevant data for a 
            correlation plot.

        Returns
        -------
        None.

        """
        super(CorrPlot, self).__init__(data)
        self.name = "correlation_plot"

    def plot(self):
        """
        Plot the correlation plot between the data and the metric.

        Returns
        -------
        None.

        """
        self.fig, self.ax = plt.subplots(
            figsize=(self.data.shape[0] * 2.3, self.data.shape[1] * 2)
        )

        p = sns.heatmap(
            self.data,
            ax=self.ax,
            linewidths=0.1,
            linecolor="white",
            cmap="YlOrRd",
        )
        self.ax.set_title(
            "Correlation matrix of the scanned parameters", self.font_dict
        )
        p.set_xticklabels(self.data, rotation=90)
        p.set_yticklabels(self.data, rotation=0)

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)


class KDEPlot(Plot):
    """Class for a 1d or 2d kernel density estimation plot."""

    def __init__(self, data, x, y=None):
        """
        Initialize the superclass Plot and the attribute name.
       
        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION.
        x : str
            Data to be used on the x-axis of the plot.
        y : str
            Column to be used on the y-axis of the plot.
            The default is None.

        Returns
        -------
        None.

        """
        super(KDEPlot, self).__init__(data)
        self.x = x
        self.y = y
        self.name = "kde_plot_" + self.x
        if y is not None:
            self.name += "-" + self.y

    def plot(self):
        """
        Plot the KDE plot.
        
        Returns
        -------
        None.

        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        if self.y is not None:
            data2 = self.data[self.y]
        else:
            data2 = None

        _ = sns.kdeplot(
            data=self.data[self.x],
            data2=data2,
            ax=self.ax,
            shade=True,
            cut=5,
            shade_lowest=False,
            cumulative=False,
            kernel="gau",
            bw="scott",
        )

        self.ax.set_title(
            "Kernel density estimator for {0} and {1}".format(self.x, self.y),
            self.font_dict,
        )

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
            tick.label.set_color("black")
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)
            tick.label.set_color("black")


class BarPlot(Plot):
    """Class for a bar plot of four parameters."""

    def __init__(self, data, x, y, hue, col):
        """
        Initialize the super-class Plot and the attribute name.

        Parameters
        ----------
        data : pd.DataFrame
        x : str
            Data to be used on the x-axis of the plot.
        y : str
            Column to be used on the y-axis of the plot.
        hue : str
            Column to be used as grouping variable in the plot.
        col : str
            Column to be used for coloration of the plot.

        Returns
        -------
        None.

        """
        super(BarPlot, self).__init__(data)
        self.x = x
        self.y = y
        self.hue = hue
        self.col = col
        self.name = (
            "bar_plot-"
            + self.x
            + "-"
            + self.y
            + "-"
            + self.hue
            + "-"
            + self.col
        )

    def plot(self):
        """
        Plot a bar plot (sns.catplot) with four parameters.
        
        Returns
        -------
        None.

        """
        # self.fig, self.ax = plt.subplots(figsize = (10,10))

        self.fig = sns.catplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.hue,
            col=self.col,
            col_wrap=4,
            legend=True,
            legend_out=True,
            size=4,
            kind="bar",
        )

        self.fig.fig.suptitle(
            "Box plot for {0}, {1}, {2}, and {3}".format(
                self.x, self.y, self.hue, self.col
            )
        )
