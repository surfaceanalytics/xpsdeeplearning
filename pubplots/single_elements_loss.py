# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:50:14 2021

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import csv

os.chdir(
    os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
)
cw = os.getcwd()

# %% Loading
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

# %% Plot metric vs. epochs
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


def plot_metric(
    history,
    metric,
    title=None,
    ylabel=None,
    zoom=False,
    zoom_x=(None, None),
    zoom_y=(None, None),
    to_file=False,
    fig_dir=None,
):
    """
    Plots the training and validation values of a metric
    against the epochs.

    Returns
    -------
    None.

    """
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
            axins.plot(
                clf_history[val_key], linewidth=3, c=colors[i], alpha=0.5
            )
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
        fig_name = os.path.join(fig_dir, "training_loss_single.png")
        fig.savefig(fig_name)


fig_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\figures"

plot_metric(
    history=history,
    metric="loss",
    title="MAE (L$_1$) loss",
    ylabel="MAE (L$_1$) loss",
    zoom=True,
    zoom_x=[100, None],
    zoom_y=[None, 0.1],
    to_file=True,
    fig_dir=fig_dir,
)
