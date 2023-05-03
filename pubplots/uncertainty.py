# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:04:45 2022

@author: pielsticker
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from string import ascii_lowercase
from matplotlib.ticker import MaxNLocator

from common import save_dir

#%%
def load_pickled_data(filepath):
    with open(filepath, "rb") as pickle_file:
        pickle_data = pickle.load(pickle_file)

    return pickle_data


folder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs\20221123_18h55m_Ni_2_classes_MC_dropout_test_predict_using_20220224_16h08m\logs"
filename = "results_synthetic.pkl"

pickle_data = load_pickled_data(os.path.join(folder, filename))

X_test = pickle_data["X_test_selected"]
X_syn = pickle_data["X_syn"]
y_test = pickle_data["y_test_selected"]
y_syn = pickle_data["y_syn"]
prob_pred_test = pickle_data["prob_pred_test_selected"]
prob_pred_syn = pickle_data["prob_pred_syn"]

energies = pickle_data["energies"]
spectrum_descriptions = pickle_data["spectrum_descriptions"]
spectrum_descriptions = [f"S3: {pickle_data['spectrum_descriptions'][0]}"]
labels = pickle_data["labels"]
indices = pickle_data["indices"]
shift_x = pickle_data["shift_x"]

#%%
def _normalize_min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def create_grid_suptitle(fig, grid, title, fontdict):
    row = fig.add_subplot(grid)
    row.set_title(f"{title}\n", fontweight="semibold", fontdict=fontdict)
    row.set_frame_on(False)
    row.axis("off")


def plot_prob_predictions_together(normalize=False):
    legend = []
    legend = [f"Test spectrum S{i+1}" for i, index in enumerate(indices)]
    legend += spectrum_descriptions

    X = np.vstack((X_test, X_syn))
    prob_pred = np.vstack((prob_pred_test, prob_pred_syn))

    fontdict = {"size": 55}

    nrows = 2
    ncols = len(labels) + 2

    fig = plt.figure(figsize=(32, 40), dpi=300)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.3)

    ax0 = fig.add_subplot(gs[0, :])
    axs = [ax0]
    for i in range(len(labels)):
        ax = fig.add_subplot(gs[1, 2 * i : 2 * i + 2])
        axs.append(ax)

    axs = np.array(axs)

    for j, label in enumerate(labels):
        axs[j + 1].set_title(
            f"({ascii_lowercase[j+1]}) {label}", fontdict=fontdict
        )
        axs[j + 1].set_xlabel("Prediction", fontdict=fontdict)
        axs[j + 1].set_ylabel("Counts", fontdict=fontdict)
        axs[j + 1].tick_params(axis="x", labelsize=fontdict["size"])
        axs[j + 1].tick_params(axis="y", labelsize=fontdict["size"])

    cmap = plt.get_cmap("tab10")
    max_counts = 0
    for i, spectrum in enumerate(X):
        color = cmap(i)
        lw = 5

        if normalize:
            axs[0].plot(
                energies, _normalize_min_max(spectrum), color=color, lw=lw
            )
        else:
            axs[0].plot(energies, spectrum, color=color, lw=lw)

        for j, row in enumerate(prob_pred[i].transpose()):
            counts, bins, patches = axs[j + 1].hist(
                row,
                bins=100,
                range=(0.0, 1.0),
                orientation="vertical",
                color=color,
                fill=True,
                linewidth=1,
                label=labels[j],
            )

            if counts.max() > max_counts:
                max_counts = counts.max()

    for ax in axs[1:]:
        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", pad=25)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 5, 10]))
        ax.set_ylim(0, max_counts + 5)

    axs[0].invert_xaxis()
    axs[0].set_xlim(np.max(energies - shift_x[0]), np.min(energies))
    axs[0].set_xlabel("Binding energy (eV)", fontdict=fontdict)
    axs[0].set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    axs[0].set_title("(a) Example spectra", fontdict=fontdict)
    axs[0].tick_params(axis="x", labelsize=fontdict["size"])
    axs[0].set_yticks([])

    axs[0].legend(
        labels=legend,
        # bbox_to_anchor=(0.9,0.9),
        # bbox_transform=plt.gcf().transFigure,
        fontsize=fontdict["size"] - 5,
    )

    create_grid_suptitle(
        fig=fig,
        grid=gs[1, :],
        title="Prediction histograms",
        fontdict=fontdict,
    )

    fig.tight_layout()

    return fig


fig = plot_prob_predictions_together(normalize=False)

for ext in [".png", ".eps"]:
    fig_path = os.path.join(save_dir, "uncertainty_ni" + ext)
    fig.savefig(fig_path, bbox_inches="tight")
