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
Plot effect of data set size on model training.
"""

import os
import csv
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common import RUNFOLDER, SAVE_DIR


def _get_total_history(csv_filepath):
    """
    Load the previous training history from the CSV log file.

    Useful for retraining.

    Returns
    -------
    history : dict
        Dictionary containing the previous training history.

    """
    history = {}
    try:
        with open(csv_filepath, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, item in row.items():
                    if key not in history.keys():
                        history[key] = []
                    history[key].append(float(item))
    except FileNotFoundError:
        pass

    return history


def load_test_loss_for_one_run(pickle_filepath):
    """
    Load the test loss after 1000 epochs.

    Returns
    -------

    """
    with open(pickle_filepath, "rb") as pickle_file:
        results = pickle.load(
            pickle_file,
        )

    return results["test_loss"]


def plot_metric(
    ax,
    history,
    metric,
    title=None,
):
    """
    Plots the training and validation values of a metric
    against the epochs.

    Returns
    -------
    None.

    """
    fontdict = {"size": 35}
    fontdict_legend = {"size": 28}

    colors = [
        "cornflowerblue",
        "red",
        "forestgreen",
        "darkgrey",
        "deeppink",
        "darkviolet",
        "orange",
    ]

    legend = []

    ax.set_xlabel("Epochs", fontdict=fontdict)
    ax.set_ylabel("MAE (L$_1$) loss", fontdict=fontdict)
    ax.tick_params(axis="x", labelsize=fontdict["size"])
    ax.tick_params(axis="y", labelsize=fontdict["size"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=4))

    ax.set_ylim(0, 0.35)

    for i, (clf_name, clf_history) in enumerate(history.items()):
        metric_history = clf_history[metric]
        ax.plot(metric_history, linewidth=3, c=colors[i])
        val_key = "val_" + metric
        ax.plot(clf_history[val_key], linewidth=3, c=colors[i], alpha=0.6)
        ax.set_xlim(-5, len(metric_history) + 5)

        legend += [f"{clf_name}k, train."]
        legend += [f"{clf_name}k, val."]

    if title:
        ax.set_title(title, fontdict=fontdict)

    ax.legend(
        legend,
        prop=fontdict_legend,
        ncol=2,
        title="Data set size",
        title_fontsize=fontdict_legend["size"],
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        shadow=False,
        mode=None,
    )


def plot_test_loss_after_1000_epochs(ax, histories, norm=False):
    """Plot the test loss of all histories after 1000 epochs."""
    fontdict = {"size": 35}
    fontdict_legend = {"size": 28}

    colors = [
        "cornflowerblue",
        "red",
        "forestgreen",
        "darkgrey",
        "deeppink",
        "darkviolet",
        "orange",
    ]

    markers = ["o", "v"]

    if norm:
        title = "(d)"
        ylabel = r"Test loss$_{1000 \; \mathrm{ep.}}$ / \n Data set size"
    else:
        title = "(c)"
        ylabel = "Test loss after 1000 epochs"

    for i, history in enumerate(histories):
        no_of_samples = list(history.keys())
        losses = []

        for sample_no, clf_history in history.items():
            test_loss = clf_history["test_loss"]

            if norm:
                test_loss /= sample_no * 1000
            losses.append(test_loss)

        ax.scatter(
            no_of_samples,
            losses,
            s=40,
            c=colors[i],
            marker=markers[i],
            linewidth=5,
        )

    ax.set_title(title, fontdict=fontdict)
    ax.set_xlabel("Data set size / 1000", fontdict=fontdict)
    ax.set_ylabel(ylabel, fontdict=fontdict)
    ax.tick_params(axis="x", labelsize=fontdict["size"])
    ax.tick_params(axis="y", labelsize=fontdict["size"])

    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fontdict["size"])

    ax.set_ylim(0, None)

    ax.legend(
        ["Ni", "Mn"],
        prop=fontdict_legend,
        title_fontsize=fontdict_legend["size"],
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        shadow=False,
        mode=None,
    )


def main():
    """Plot effect of data set size on model training."""

    classifiers_Ni = {
        25: "20230427_14h58m_Ni_linear_combination_normalized_inputs_small_gas_phase_25k",
        50: "20230427_20h37m_Ni_linear_combination_normalized_inputs_small_gas_phase_50k",
        100: "20230427_20h39m_Ni_linear_combination_normalized_inputs_small_gas_phase_100k",
        150: "20230427_20h42m_Ni_linear_combination_normalized_inputs_small_gas_phase_150k",
        200: "20230427_20h43m_Ni_linear_combination_normalized_inputs_small_gas_phase_200k",
        250: "20230427_20h45m_Ni_linear_combination_normalized_inputs_small_gas_phase_250k",
    }

    classifiers_Mn = {
        25: "20230502_09h44m_Mn_linear_combination_normalized_inputs_small_gas_phase_25k",
        50: "20230428_11h19m_Mn_linear_combination_normalized_inputs_small_gas_phase_50k",
        100: "20230502_09h33m_Mn_linear_combination_normalized_inputs_small_gas_phase_100k",
        150: "20230502_09h55m_Mn_linear_combination_normalized_inputs_small_gas_phase_150k",
        200: "20230502_10h23m_Mn_linear_combination_normalized_inputs_small_gas_phase_200k",
        250: "20230427_20h48m_Mn_linear_combination_normalized_inputs_small_gas_phase_250k",
    }

    history_Ni = {}
    history_Mn = {}

    for clf_name, clf_path in classifiers_Ni.items():
        logpath = os.path.join(*[RUNFOLDER, clf_path, "logs/log.csv"])
        history_Ni[clf_name] = _get_total_history(logpath)
        pkl_path = os.path.join(*[RUNFOLDER, clf_path, "logs", "results.pkl"])
        test_loss = load_test_loss_for_one_run(pkl_path)
        history_Ni[clf_name]["test_loss"] = test_loss
    for clf_name, clf_path in classifiers_Mn.items():
        logpath = os.path.join(*[RUNFOLDER, clf_path, "logs/log.csv"])
        history_Mn[clf_name] = _get_total_history(logpath)
        pkl_path = os.path.join(*[RUNFOLDER, clf_path, "logs", "results.pkl"])
        test_loss = load_test_loss_for_one_run(pkl_path)
        history_Mn[clf_name]["test_loss"] = test_loss

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(30, 22),
        gridspec_kw={"hspace": 0.3, "wspace": 0.3},
        dpi=300,
    )

    metric = "loss"
    titles = ["(a) Ni", "(b) Mn"]

    histories = [history_Ni, history_Mn]

    for j, hist_dict in enumerate(histories):
        plot_metric(
            axs[0, j],
            hist_dict,
            metric,
            title=titles[j],
        )

    plot_test_loss_after_1000_epochs(axs[1, 0], histories, norm=False)
    plot_test_loss_after_1000_epochs(axs[1, 1], histories, norm=True)

    fig.tight_layout()
    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "sample_size_effect" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
