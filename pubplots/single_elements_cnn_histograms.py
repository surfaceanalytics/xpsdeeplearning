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
Histogram of predictions on XPS data with single elements.
"""

import os
import matplotlib.pyplot as plt
import pickle
from matplotlib import gridspec
from sklearn.metrics import mean_absolute_error
import numpy as np

from common import RUNFOLDER, SAVE_DIR


class Wrapper:
    """Wrapper for loading and plotting."""

    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.fontdict = {"size": 25}
        self.fontdict_small = {"size": 14}
        self.fontdict_legend = {"size": 16}

        self.title_dict = {
            "Co": "(a) Co 2p",
            "Cu": "(b) Cu 2p",
            "Fe": "(c) Fe 2p",
            "Mn": "(d) Mn 2p",
            "Ni": "(e) Ni 2p",
            "Pd": "(f) Pd 3d",
            "Ti": "(g) Ti 2p",
        }

    def _load_predictions_for_one_run(self, pickle_filepath):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------

        """
        with open(pickle_filepath, "rb") as pickle_file:
            results = pickle.load(
                pickle_file,
            )

        y_test = results["y_test"]
        pred_test = results["pred_test"]

        return y_test, pred_test

    def _calculate_test_losses_for_one_run(self, loss_func, y_test, pred_test):
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
        losses_test = np.array(
            [loss_func(y_test[i], pred_test[i]) for i in range(y_test.shape[0])]
        )

        return losses_test

    def load_predictions(self, classifiers):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------

        """
        self.results = {}
        for clf, clf_name in classifiers.items():
            pkl_path = os.path.join(*[self.datafolder, clf_name, "logs", "results.pkl"])

            y_test, pred_test = self._load_predictions_for_one_run(pkl_path)

            self.results[clf] = {"y_test": y_test, "pred_test": pred_test}

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
        print("Calculating loss for each example...")
        for clf, result in self.results.items():
            y_test = result["y_test"]
            pred_test = result["pred_test"]
            losses_test = self._calculate_test_losses_for_one_run(
                loss_func, y_test, pred_test
            )
            result["losses_test"] = losses_test
            print(
                clf,
                np.median(losses_test),
                np.percentile(losses_test, [25, 75]),
            )
        print("Done!")

    def _add_loss_histogram(self, ax, losses_test, title):
        """
        Plots a histogram of the lossses for each example in the pred_test
        data set.

        Returns
        -------
        None.

        """
        ax.set_title(title, fontdict=self.fontdict)

        ax.set_xlabel("Mean Absolute Error ", fontdict=self.fontdict)
        ax.set_ylabel("Counts", fontdict=self.fontdict)
        ax.tick_params(axis="x", labelsize=self.fontdict["size"])
        ax.tick_params(axis="y", labelsize=self.fontdict["size"])

        N, _, hist_patches = ax.hist(
            losses_test,
            bins=100,
            range=(0.0, 0.5),
            histtype="bar",
            fill=False,
            cumulative=False,
        )

        thresholds = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
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

            try:
                if float(quantification[-1][1].split(" ")[0]) < 100.0:
                    quantification.append([t, f"{ng_p} %", f"{nb_p} %"])
            except IndexError:
                quantification.append([t, f"{ng_p} %", f"{nb_p} %"])

        col_labels = [
            "MAE threshold",
            "% correct",
            "% wrong",
        ]

        field_height = 0.1
        y0 = 0.975 - field_height * len(quantification)

        table = ax.table(
            cellText=quantification,
            cellLoc="center",
            colLabels=col_labels,
            bbox=[0.35, y0, 0.625, field_height * len(quantification)],
            zorder=500,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        ax.set_xlim(hist_patches[0].xy[0], 0.4)
        ax.tick_params(axis="x", pad=12)

        return ax

    def plot_all(self):
        """Plot histograms."""
        self.x_max = {
            "Co": 0.25,
            "Cu": 0.25,
            "Fe": 0.25,
            "Mn": 0.4,
            "Ni": 0.25,
            "Pd": 0.25,
            "Ti": 0.25,
        }
        ncols = 2

        self.fig = plt.figure(figsize=(16, 24), dpi=300)
        gs = gridspec.GridSpec(nrows=4, ncols=2 * ncols, wspace=0.2, hspace=0.3)

        # Set up 2 plots in first three row and 1 in last row.
        ax0_0 = self.fig.add_subplot(gs[0, :2])
        ax0_1 = self.fig.add_subplot(gs[0, 2:])
        ax1_0 = self.fig.add_subplot(gs[1, :2])
        ax1_1 = self.fig.add_subplot(gs[1, 2:])
        ax2_0 = self.fig.add_subplot(gs[2, :2])
        ax2_1 = self.fig.add_subplot(gs[2, 2:])
        ax3 = self.fig.add_subplot(gs[3, 1:3])

        self.axs = np.array(
            [[ax0_0, ax0_1], [ax1_0, ax1_1], [ax2_0, ax2_1], [ax3, None]]
        )

        for i, clf in enumerate(self.results.keys()):
            row, col = int(i / ncols), i % ncols
            losses_test = self.results[clf]["losses_test"]
            title = self.title_dict[clf]

            self.axs[row, col] = self._add_loss_histogram(
                ax=self.axs[row, col], losses_test=losses_test, title=title
            )

        gs.tight_layout(self.fig)

        return self.fig


def main():
    """Plot histogram of predictions on XPS data with single elements."""
    classifiers = {
        "Co": "20220628_08h58m_Co_linear_combination_normalized_inputs_small_gas_phase",
        "Cu": "20220628_09h58m_Cu_linear_combination_normalized_inputs_small_gas_phase",
        "Fe": "20220629_09h36m_Fe_linear_combination_normalized_inputs_small_gas_phase",
        "Mn": "20220628_11h57m_Mn_linear_combination_normalized_inputs_small_gas_phase",
        "Ni": "20220627_16h49m_Ni_linear_combination_normalized_inputs_small_gas_phase",
        "Pd": "20220627_16h50m_Pd_linear_combination_normalized_inputs_small_gas_phase",
        "Ti": "20220628_11h55m_Ti_linear_combination_normalized_inputs_small_gas_phase",
    }

    wrapper = Wrapper(RUNFOLDER)
    wrapper.load_predictions(classifiers)
    wrapper.calculate_test_losses(loss_func=mean_absolute_error)
    # results_maae = wrapper.calculate_test_losses(loss_func=maximum_absolute_error)
    fig = wrapper.plot_all()
    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "hist_cnn_single" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
