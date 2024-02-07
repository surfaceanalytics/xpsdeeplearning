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
Plot visualization of window fitting.
"""

import os
import csv
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import mean_absolute_error
import numpy as np

from common import (
    ParserWrapper,
    print_mae_info,
    maximum_absolute_error,
    save_dir,
)


class Wrapper(ParserWrapper):
    """Parser for XPS data stored in TXT files."""

    def __init__(self, datafolder: str, file_dict: dict):
        """
        Initialize empty data dictionary.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        super(Wrapper, self).__init__(datafolder=datafolder, file_dict=file_dict)
        self.fontdict = {"size": 25}
        self.fontdict_small = {"size": 14}
        self.fontdict_legend = {"size": 16}

        self.history = {}
        self.y_test = np.array([])
        self.pred_test = np.array([])
        self.fig = plt.figure(figsize=(24, 16), dpi=300)
        self.axs = np.array([])

        self.color_dict = {
            "CPS": "black",
            "window": "orangered",
            "train_loss": "cornflowerblue",
            "val_loss": "forestgreen",
        }

    def load_history(self, csv_filepath):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing the previous training history.

        """
        history = {}
        with open(csv_filepath, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, item in row.items():
                    if key not in history.keys():
                        history[key] = []
                    history[key].append(float(item))

        self.history = history

        return self.history

    def _add_metric_plot(self, ax, history, metric, title=None, ylabel=None):
        """
        Plots the training and validation values of a metric
        against the epochs.

        Returns
        -------
        None.

        """
        fontdict = {"size": 25}
        fontdict_legend = {"size": 21}

        metric_cap = metric.capitalize()
        legend = ["Training", "Validation"]

        if ylabel is None:
            ylabel = metric_cap
        ax.set_ylabel(str(ylabel), fontdict=fontdict)

        ax.set_title("(b) Training and validation loss", fontdict=fontdict)

        ax.set_xlabel("Epochs", fontdict=fontdict)
        ax.tick_params(axis="x", labelsize=fontdict["size"])
        ax.tick_params(axis="y", labelsize=fontdict["size"])

        ax.plot(self.history[metric], linewidth=3, c=self.color_dict["train_loss"])
        val_key = "val_" + metric
        ax.plot(
            self.history[val_key],
            linewidth=3,
            c=self.color_dict["val_loss"],
            alpha=0.6,
        )
        ax.set_xlim(-5, len(self.history[metric]) + 5)

        ax.set_ylabel(str(ylabel), fontdict=fontdict)

        ax.legend(
            legend,
            title="MAE (L$_1$) loss",
            prop=fontdict_legend,
            title_fontsize=fontdict_legend["size"],
            bbox_to_anchor=(0, 0, 1, 1),
            fancybox=True,
            shadow=False,
            mode=None,
        )

        return ax

    def load_predictions(self, pickle_filepath):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------

        """
        with open(pickle_filepath, "rb") as pickle_file:
            results = pickle.load(
                pickle_file,
                # encoding="utf8"
            )
        self.y_test = results["y_test"]
        self.pred_test = results["pred_test"]

        return self.y_test, self.pred_test

    def calculate_test_losses(self, loss_func):
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
        self.losses_test = np.array(
            [
                loss_func(self.y_test[i], self.pred_test[i])
                for i in range(self.y_test.shape[0])
            ]
        )

        return self.losses_test

    def _add_loss_histogram(
        self,
        ax,
        losses_test,
    ):
        """
        Plots a histogram of the lossses for each example in the pred_test
        data set.

        Returns
        -------
        None.

        """

        ax.set_title("(d) MAEs of test set", fontdict=self.fontdict)

        ax.set_xlabel("Mean Absolute Error ", fontdict=self.fontdict)
        ax.set_ylabel("Counts", fontdict=self.fontdict)
        ax.tick_params(axis="x", labelsize=self.fontdict["size"])
        ax.tick_params(axis="y", labelsize=self.fontdict["size"])

        N, bins, hist_patches = ax.hist(
            losses_test,
            bins=100,
            histtype="bar",
            fill=False,
            cumulative=False,
        )

        thresholds = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
        quantification = []

        for t in thresholds:
            ng, nb = 0, 0
            for i, patch in enumerate(hist_patches):
                if patch.xy[0] < t:
                    ng += N[i]
                else:
                    nb += N[i]
            ng_p = np.round(ng / (ng + nb) * 100, 1)
            nb_p = np.round(nb / (ng + nb) * 100, 1)

            quantification.append([t, f"{ng_p} %", f"{nb_p} %"])

        col_labels = [
            "MAE threshold",
            "% of correct \n quantications",
            "% of wrong \n quantications",
        ]

        table = ax.table(
            cellText=quantification,
            cellLoc="center",
            colLabels=col_labels,
            bbox=[0.38, 0.4, 0.6, 0.48],
            zorder=500,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        ax.set_xlim(
            hist_patches[0].xy[0],
            hist_patches[-1].xy[0],
        )

        return ax

    def plot_all(self, with_fits=False):
        gs = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            wspace=0.2,
            hspace=0.3,
            width_ratios=[1.5, 1],
        )

        # Set up 2 plots in both rows.
        ax0_0 = self.fig.add_subplot(gs[0, 0])
        ax0_1 = self.fig.add_subplot(gs[0, 1])
        ax1_0 = self.fig.add_subplot(gs[1, 0])
        ax1_1 = self.fig.add_subplot(gs[1, 1])
        self.axs = np.array([[ax0_0, ax0_1], [ax1_0, ax1_1]])

        self.axs[0, 1] = self._add_metric_plot(
            ax=self.axs[0, 1],
            history=self.history,
            metric="loss",
            title="MAE loss",
            ylabel="MAE loss",
        )

        self.axs[1, 1] = self._add_loss_histogram(
            ax=self.axs[1, 1],
            losses_test=self.losses_test,
        )

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            self.axs[i, 0].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            self.axs[i, 0].set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
            self.axs[i, 0].tick_params(axis="x", labelsize=self.fontdict["size"])
            self.axs[i, 0].set_yticklabels([])
            self.axs[i, 0].tick_params(axis="y", which="both", right=False, left=False)

            if not with_fits:
                parser.header_names = ["CPS"]

            for j, header_name in enumerate(parser.header_names):
                for spectrum_name in parser.names:
                    if spectrum_name in header_name:
                        name = spectrum_name

                color = self.color_dict[name]

                start = parser.fit_start
                end = parser.fit_end

                try:
                    self.axs[i, 0].plot(x[start:end], y[start:end, j], c=color)
                except IndexError:
                    self.axs[i, 0].plot(x[start:end], y[start:end], c=color)
            if i == 0:
                text = "Ground truth:"
            else:
                text = "Quantification:"

            self.axs[i, 0].set_title(parser.title, fontdict=self.fontdict)

            quantification = [f"{p} %" for p in list(parser.quantification.values())]

            keys = list(parser.quantification.keys())
            col_labels = self._reformat_label_list(keys)

            table = self.axs[i, 0].table(
                cellText=[quantification],
                cellLoc="center",
                colLabels=col_labels,
                bbox=[0.03, 0.04, 0.095 * len(quantification), 0.15],
                zorder=500,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.fontdict_small["size"])

            self.axs[i, 0].text(
                0.03,
                0.22,
                text,
                horizontalalignment="left",
                size=self.fontdict_small["size"],
                verticalalignment="center",
                transform=self.axs[i, 0].transAxes,
            )

        window = [705, 805]

        self.axs[0, 0].set_xlim(left=np.max(x), right=np.min(x))
        self.axs[1, 0].set_xlim(left=window[1], right=window[0])

        ymin, ymax = self.axs[0, 0].get_ylim()

        rect = patches.Rectangle(
            (window[0], 0.27 * ymax),
            window[1] - window[0],
            0.71 * ymax,
            linewidth=2,
            edgecolor=self.color_dict["window"],
            facecolor="none",
        )
        self.axs[0, 0].add_patch(rect)

        arrow = patches.FancyArrowPatch(
            (window[0], 0.65 * ymax),
            (window[1], 0.65 * ymax),
            arrowstyle="<->",
            mutation_scale=20,
            color=self.color_dict["window"],
        )
        self.axs[0, 0].add_patch(arrow)

        self.axs[0, 0].text(
            (window[1] - window[0]) / 2 + window[0],
            0.7 * ymax,
            "100 eV window",
            horizontalalignment="center",
            size=self.fontdict["size"],
            color=self.color_dict["window"],
            verticalalignment="center",
        )

        texts = [
            ("Co 2p", 0.26, 0.9),
            ("Fe 2p", 0.79, 0.47),
        ]

        for t in texts:
            self.axs[0, 0].text(
                x=t[1],
                y=t[2],
                s=t[0],
                horizontalalignment="left",
                size=self.fontdict["size"],
                verticalalignment="center",
                transform=self.axs[0, 0].transAxes,
            )

        for w in window:
            con = ConnectionPatch(
                xyA=(w, 0.27 * ymax),
                xyB=(w, self.axs[1, 0].get_ylim()[1]),
                coordsA="data",
                coordsB="data",
                axesA=self.axs[0, 0],
                axesB=self.axs[1, 0],
                linestyle="dashed",
                arrowstyle="-",
                linewidth=2,
                color=self.color_dict["window"],
                alpha=0.5,
            )
            self.axs[1, 0].add_artist(con)
        self.axs[0, 0].set_axisbelow(False)

        # MAE of test spectrum
        y_true = np.array(list(self.parsers[0].quantification.values())) / 100
        y_pred = np.array(list(self.parsers[1].quantification.values())) / 100
        mae_single = f"MAE: {np.round(mean_absolute_error(y_true, y_pred),2)}"
        self.axs[1, 0].text(
            x=0.75,
            y=0.125,
            s=mae_single,
            horizontalalignment="left",
            size=self.fontdict["size"],
            verticalalignment="center",
            transform=self.axs[1, 0].transAxes,
        )

        # gs.tight_layout(self.fig)

        return self.fig, self.axs


def print_diff(losses, threshold):
    correct = 0
    wrong = 0
    for loss in losses:
        if loss < threshold:
            correct += 1
        else:
            wrong += 1
    print(
        f"No. of correct classifications {correct}/{len(losses)} = {correct/len(losses)*100} %"
    )
    print(
        f"No. of wrong classifications {wrong}/{len(losses)} = {wrong/len(losses)*100} %"
    )


def main():
    """Plot visualization of window fitting."""
    datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
    clfpath = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs\20210914_19h13m_FeCo_combined_without_auger_7_classes_100eV_window\logs"
    logpath = os.path.join(clfpath, "log.csv")
    predpath = os.path.join(clfpath, "results.pkl")

    file_dict = {
        "true": {
            "filename": "cofe_sim.txt",
            "title": "(a) Simulated spectrum with window",
            "fit_start": 0,
            "fit_end": -1,
            "shift_x": 0.0,
            "noise": 32.567,
            "FWHM": 1.28,
            "scatterer": "N2",
            "distance": 0.18,
            "pressure": 0.3,
            "quantification": {
                "Co metal": 26.7,
                "CoO": 22.2,
                "Co3O4": 24.0,
                "Fe metal": 0.0,
                "FeO": 14.4,
                "Fe3O4": 12.7,
                "Fe2O3": 0.0,
            },
        },
        "neural_net": {
            "filename": "cofe_sim.txt",
            "title": "(c) Neural network quantifcation on one example",
            "fit_start": 0,
            "fit_end": -1,
            "shift_x": 0.0,
            "quantification": {
                "Co metal": 25.7,
                "CoO": 22.8,
                "Co3O4": 28.3,
                "Fe metal": 4.1,
                "FeO": 10.5,
                "Fe3O4": 5.8,
                "Fe2O3": 2.8,
            },
        },
    }

    wrapper = Wrapper(datafolder, file_dict)
    wrapper.parse_data(bg=True, envelope=True)
    history = wrapper.load_history(logpath)
    y_test, pred_test = wrapper.load_predictions(predpath)
    losses_test = wrapper.calculate_test_losses(loss_func=mean_absolute_error)

    fig, ax = wrapper.plot_all()
    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(save_dir, "window" + ext)
        fig.savefig(fig_path, bbox_inches="tight")

    print_mae_info(losses_test, "Window", precision=4)

    maae_test = wrapper.calculate_test_losses(loss_func=maximum_absolute_error)

    threshold = 0.1  # percent
    losses = {"MAE": losses_test, "MaAE": maae_test}

    for name, losses in losses.items():
        print(name + ":")
        print_diff(losses, threshold)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
