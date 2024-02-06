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
Plot training loss for models trained on artificial data sets of
different elements.
"""

import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from common import save_dir


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


def plot_metric(
    history,
    metric,
    title=None,
    ylabel=None,
    zoom=False,
    zoom_x=(None, None),
    zoom_y=(None, None),
    to_file=False,
):
    """
    Plots the training and validation values of a metric
    against the epochs.

    Returns
    -------
    None.

    """
    fontdict = {"size": 25}
    fontdict_small = {"size": 20}
    fontdict_legend = {"size": 17}

    colors = [
        "cornflowerblue",
        "red",
        "forestgreen",
        "darkgrey",
        "deeppink",
        "darkviolet",
        "orange",
    ]

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

    legend = []

    metric_cap = metric.capitalize()

    if ylabel is None:
        ylabel = metric_cap
    ax.set_ylabel(str(ylabel), fontdict=fontdict)

    ax.set_xlabel("Epochs", fontdict=fontdict)
    ax.tick_params(axis="x", labelsize=fontdict["size"])
    ax.tick_params(axis="y", labelsize=fontdict["size"])

    if zoom:
        axins = inset_axes(
            ax,
            width=6,
            height=3,
            loc=2,
            bbox_to_anchor=(0.4, 0.7),
            bbox_transform=ax.figure.transFigure,
        )
        axins.set_xlabel("Epochs", fontdict=fontdict_small)
        axins.set_ylabel(str(ylabel), fontdict=fontdict_small)
        axins.tick_params(axis="x", labelsize=fontdict_small["size"])
        axins.tick_params(axis="y", labelsize=fontdict_small["size"])

    for i, (clf_name, clf_history) in enumerate(history.items()):
        metric_history = clf_history[metric]
        ax.plot(metric_history, linewidth=3, c=colors[i])
        val_key = "val_" + metric
        ax.plot(clf_history[val_key], linewidth=3, c=colors[i], alpha=0.6)
        ax.set_xlim(-5, len(metric_history) + 5)

        legend += [f"{clf_name}, Training"]
        legend += [f"{clf_name}, Validation"]

        if zoom:
            axins.plot(metric_history, linewidth=3, c=colors[i])
            axins.plot(clf_history[val_key], linewidth=3, c=colors[i], alpha=0.5)
            if zoom_x[1] is None:
                zoom_end = len(metric_history) + 5
            else:
                zoom_end = zoom_x[1]
            axins.set_xlim(zoom_x[0], zoom_end)  # apply the x-limits
            axins.set_ylim(zoom_y[0], zoom_y[1])
            # mark_inset(ax, axins, loc1=2, loc2=4, ec="0.5")

    # Only for a better legend.
    ax.plot(np.zeros(1), np.zeros([1, 3]), color="w", alpha=0, label=" ")
    ax.plot(np.zeros(1), np.zeros([1, 3]), color="w", alpha=0, label=" ")
    legend.append("")
    legend.append("")

    ax.legend(
        legend,
        prop=fontdict_legend,
        ncol=4,
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        shadow=False,
        mode=None,
    )

    plt.show()

    if to_file:
        # fig_name = os.path.join(fig_dir, f"{metric}.png")
        for ext in [".png", ".eps"]:
            fig_path = os.path.join(save_dir, "training_loss_single" + ext)
            fig.savefig(fig_path, bbox_inches="tight")


def main():
    """
    Plot training loss for models trained on artificial data sets of
    different elements.
    """
    input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs"

    classifiers = {
        "Co": "20220628_08h58m_Co_linear_combination_normalized_inputs_small_gas_phase",
        "Cu": "20220628_09h58m_Cu_linear_combination_normalized_inputs_small_gas_phase",
        "Fe": "20220629_09h36m_Fe_linear_combination_normalized_inputs_small_gas_phase",
        # "Fe_1000": "20220629_09h36m_Fe_linear_combination_normalized_inputs_small_gas_phase_after_1000_epochs",
        "Mn": "20220628_11h57m_Mn_linear_combination_normalized_inputs_small_gas_phase",
        # "Mn_1000": "20220628_11h57m_Mn_linear_combination_normalized_inputs_small_gas_phase_after_1000_epochs",
        "Ni": "20220627_16h49m_Ni_linear_combination_normalized_inputs_small_gas_phase",
        "Pd": "20220627_16h50m_Pd_linear_combination_normalized_inputs_small_gas_phase",
        "Ti": "20220628_11h55m_Ti_linear_combination_normalized_inputs_small_gas_phase",
        # "CoFe": "20210914_19h11m_FeCo_combined_without_auger_7_classes_no_window",
        # "NiCoFe": "20210604_23h09m_NiCoFe_9_classes_long_linear_comb_small_gas_phase",
    }

    history = {}

    for clf_name, clf_path in classifiers.items():
        logpath = os.path.join(*[input_datafolder, clf_path, "logs/log.csv"])
        history[clf_name] = _get_total_history(logpath)

    #  Plot metric vs. epochs
    plot_metric(
        history=history,
        metric="loss",
        title="MAE (L$_1$) loss",
        ylabel="MAE (L$_1$) loss",
        zoom=True,
        zoom_x=[100, None],
        zoom_y=[None, 0.1],
        to_file=True,
    )


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
