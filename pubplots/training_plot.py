# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:50:14 2021

@author: pielsticker
"""

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

os.chdir(
    os.path.join(
        os.path.abspath(__file__).split("deepxps")[0], "deepxps"
    )
)
cw = os.getcwd()

# noqa: E402
from xpsdeeplearning.simulation.base_model.spectra import (
    MeasuredSpectrum,
    SimulatedSpectrum,
)
from xpsdeeplearning.simulation.base_model.figures import Figure
from xpsdeeplearning.simulation.sim import Simulation


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
    "Co": "20220627_09h30m_Co_linear_combination_normalized_inputs_no_scattering",
    "Cu": "20220627_09h30m_Cu_linear_combination_normalized_inputs_no_scattering",
    "Fe": "20220627_13h45m_Fe_linear_combination_normalized_inputs_no_scattering",
    "Mn": "20220627_15h17m_Mn_linear_combination_normalized_inputs_no_scattering",
    "Ni": "20220627_09h35m_Ni_linear_combination_normalized_inputs_no_scattering",
    "Pd": "20220627_09h36m_Pd_linear_combination_normalized_inputs_no_scattering",
    "Ti": "20220627_09h39m_Ti_linear_combination_normalized_inputs_no_scattering"
    }

classifiers_small_gas_phase = {
    "Co": "20220628_08h58m_Co_linear_combination_normalized_inputs_small_gas_phase",
    "Cu": "20220628_09h58m_Cu_linear_combination_normalized_inputs_small_gas_phase",
    "Fe": "20220629_09h36m_Fe_linear_combination_normalized_inputs_small_gas_phase",
    "Mn": "20220628_11h57m_Mn_linear_combination_normalized_inputs_small_gas_phase",
    "Ni": "20220627_16h49m_Ni_linear_combination_normalized_inputs_small_gas_phase",
    "Pd": "20220627_16h50m_Pd_linear_combination_normalized_inputs_small_gas_phase",
    "Ti": "20220628_11h55m_Ti_linear_combination_normalized_inputs_small_gas_phase"
    }

classifiers_big_gas_phase = {
    "Co": "",
    "Cu": "",
    "Fe": "",
    "Mn": "",
    "Ni": "",
    "Pd": "",
    "Ti": ""
    }

history = {
    "Co": [],
    "Cu": [],
    "Fe": [],
    "Mn": [],
    "Ni": [],
    "Pd": [],
    "Ti": []
    }
# =============================================================================
# for clf_name, clf_path in classifiers.items():
#     logpath = os.path.join(*[input_datafolder, clf_path, "logs/log.csv"])
#     history[clf_name] += [_get_total_history(logpath)]
# =============================================================================
for clf_name, clf_path in classifiers_small_gas_phase.items():
    logpath = os.path.join(*[input_datafolder, clf_path, "logs/log.csv"])
    history[clf_name] += [_get_total_history(logpath)]
# =============================================================================
# for clf_name, clf_path in classifiers_big_gas_phase.items():
#     logpath = os.path.join(*[input_datafolder, clf_path, "logs/log.csv"])
#     history[clf_name] += [_get_total_history(logpath)]
# =============================================================================

# %% Plot metric vs. epochs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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
        fig_dir=None
    ):
    """
    Plots the training and validation values of a metric
    against the epochs.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(15,10), dpi=300)

    metric_cap = metric.capitalize()
    legend = []

    metric_cap = metric.capitalize()
# =============================================================================
#     if title is None:
#         title = metric_cap
#     ax.set_title(str(title), fontdict=fontdict)
# =============================================================================
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
            bbox_to_anchor=(.4, 0.7),
            bbox_transform=ax.figure.transFigure)
        axins.set_xlabel("Epochs", fontdict=fontdict_small)
        axins.set_ylabel(str(ylabel), fontdict=fontdict_small)
        axins.tick_params(axis="x", labelsize=fontdict_small["size"])
        axins.tick_params(axis="y", labelsize=fontdict_small["size"])

    for i, (clf_name, clf_history) in enumerate(history.items()):
        for j, hist in enumerate(clf_history):

            metric_history = hist[metric]
            ax.plot(metric_history, linewidth=3, c=colors[i+j])
            val_key = "val_" + metric
            ax.plot(hist[val_key], linewidth=3, c=colors[i+j], alpha=0.6)
            ax.set_xlim(-5, len(metric_history)+5)

            if j == 1:
                t = f"no scatter"
            elif j == 0:
                t = f"small gas phase"
            elif j == 2:
                t = "big gas phase"
            legend += [f"{clf_name}, Train"]#", {t}: Train"]
            legend += [f"{clf_name}, Validation"]#, {t}: Validation"]

            if zoom:
                axins.plot(metric_history, linewidth=3, c=colors[i+j])
                axins.plot(hist[val_key], linewidth=3, c=colors[i+j], alpha=0.5)
                if zoom_x[1] is None:
                   zoom_x[1] = len(metric_history) + 5
                axins.set_xlim(zoom_x[0], zoom_x[1]) # apply the x-limits
                axins.set_ylim(zoom_y[0], zoom_y[1])
             #   mark_inset(ax, axins, loc1=2, loc2=4, ec="0.5")

    # Only for a better legend.
    ax.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
    ax.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
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



    if to_file:
        fig_name = os.path.join(fig_dir, f"{metric}.png")
        fig.savefig(fig_name)


plot_metric(
    history=history,
    metric="loss",
    title="MAE (L$_1$) loss",
    ylabel="MAE (L$_1$) loss",
    zoom=True,
    zoom_x=[100, None],
    zoom_y=[None, 0.1],
    to_file=True,
    fig_dir = r"C:\Users\pielsticker\Downloads"
    )


