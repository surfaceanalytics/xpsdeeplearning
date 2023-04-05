# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:22:27 2022

@author: pielsticker
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

from common import get_xlsxpath, print_mae_info, maximum_absolute_error


os.chdir(
    os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
)

input_datafolder = os.path.join(os.getcwd(), "utils")
fit_datafolder = os.path.join(input_datafolder, "fit_comparison")

# %%
def print_for_one_spectrum(number, method):
    if method == "biesinger":
        df = df_biesinger
        mae = mae_biesinger
    elif method == "fit_model":
        df = df_fit_model
        mae = mae_fit_model
    elif method == "lineshapes":
        df = df_lineshapes
        mae = mae_lineshapes
    elif method == "nn":
        df = df_nn
        mae = mae_nn

    number_filled = str(number).zfill(2)
    spectrum_text = f"Synthetic spectrum no. {number_filled}"
    y_true = df_true.loc[spectrum_text]
    y_pred = df.loc[spectrum_text]
    mae_one = mae[number]

    print(
        f"\t True: \n {y_true}\n ",
        f"\t {method.capitalize()}: \n {y_pred} \n",
        f"\t MAE = {mae_one}",
    )


def _add_loss_histogram(ax, mae, title):
    fontdict = {"size": 40}
    ax.set_title(title, fontdict=fontdict, multialignment="center")

    ax.set_xlabel("Mean Absolute Error ", fontdict=fontdict)
    ax.set_ylabel("Counts", fontdict=fontdict)
    ax.tick_params(axis="x", labelsize=fontdict["size"])
    ax.tick_params(axis="y", labelsize=fontdict["size"])

    N, bins, hist_patches = ax.hist(
        mae,
        bins=30,
        range=(0, 0.5),
        histtype="bar",
        fill=False,
        cumulative=False,
    )

    ax.text(
        0.7,
        0.825,
        f"Median MAE:\n{np.round(np.median(mae),2)}",
        horizontalalignment="center",
        size=fontdict["size"],
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.set_xlim(hist_patches[0].xy[0], 0.5)  # hist_patches[-1].xy[0],)
    ax.tick_params(axis="x", pad=12)

    return ax


#%%
cols = ["Fe metal", "FeO", "Fe3O4", "Fe2O3"]

df_true = pd.read_csv(
    get_xlsxpath(fit_datafolder, "truth"), index_col=0, sep=";"
)

df_biesinger = pd.read_csv(
    get_xlsxpath(fit_datafolder, "biesinger"), index_col=0, sep=";"
)
df_biesinger = df_biesinger.divide(100)
df_biesinger.columns = df_biesinger.columns.str.replace(" %", "")
df_biesinger["Fe3O4"] = df_biesinger["Fe3O4 2+"] + df_biesinger["Fe3O4 3+"]
df_biesinger = df_biesinger.drop(columns=["Fe3O4 2+", "Fe3O4 3+", "Fe sat"])
df_biesinger = df_biesinger[cols]
df_biesinger = df_biesinger.div(df_biesinger.sum(axis=1), axis=0)
mae_biesinger = mean_absolute_error(
    df_biesinger.to_numpy().T, df_true.to_numpy().T, multioutput="raw_values"
)
maae_biesinger = maximum_absolute_error(
    df_biesinger.to_numpy().T, df_true.to_numpy().T
)
df_biesinger["MAE"] = mae_biesinger
df_biesinger["MaAE"] = maae_biesinger
print_mae_info(mae_biesinger, "Biesinger")


df_fit_model = pd.read_csv(
    get_xlsxpath(fit_datafolder, "fit_model"), index_col=0, sep=";"
)
df_fit_model = df_fit_model.divide(100)
df_fit_model.columns = df_fit_model.columns.str.replace(" %", "")
df_fit_model = df_fit_model[cols]
fe3o4_indices = df_true.loc[(df_true["Fe3O4"] == 1.0)].index
df_fit_model.loc[fe3o4_indices]
mae_fit_model = mean_absolute_error(
    df_fit_model.to_numpy().T, df_true.to_numpy().T, multioutput="raw_values"
)
maae_fit_model = maximum_absolute_error(
    df_fit_model.to_numpy().T, df_true.to_numpy().T
)
df_fit_model["MAE"] = mae_fit_model
df_fit_model["MaAE"] = maae_fit_model
print_mae_info(mae_fit_model, "Fit model")


df_lineshapes = pd.read_csv(
    get_xlsxpath(fit_datafolder, "lineshapes"), index_col=0, sep=";"
)
df_lineshapes = df_lineshapes.divide(100)
df_lineshapes.columns = df_lineshapes.columns.str.replace(" %", "")
df_lineshapes = df_lineshapes[cols]
mae_lineshapes = mean_absolute_error(
    df_lineshapes.to_numpy().T, df_true.to_numpy().T, multioutput="raw_values"
)
maae_lineshapes = maximum_absolute_error(
    df_lineshapes.to_numpy().T, df_true.to_numpy().T
)
df_lineshapes["MAE"] = mae_lineshapes
df_lineshapes["MaAE"] = maae_lineshapes
print_mae_info(mae_lineshapes, "lineshapes")


df_nn = pd.read_csv(get_xlsxpath(fit_datafolder, "nn"), index_col=0, sep=";")
mae_nn = mean_absolute_error(
    df_nn.to_numpy().T, df_true.to_numpy().T, multioutput="raw_values"
)
maae_nn = maximum_absolute_error(df_nn.to_numpy().T, df_true.to_numpy().T)
df_nn["MAE"] = mae_nn
df_nn["MaAE"] = maae_nn
print_mae_info(mae_nn, "Neural network")

fig, axs = plt.subplots(
    nrows=1,
    ncols=4,
    squeeze=False,
    figsize=(40, 16),
    gridspec_kw={"wspace": 0.3},
    dpi=300,
)

axs[0, 0] = _add_loss_histogram(
    axs[0, 0],
    mae_biesinger,
    title="(a) M1: Fit according to \n Grosvenor et al.",
)
axs[0, 1] = _add_loss_histogram(
    axs[0, 1], mae_fit_model, title="(b) M2: Manual peak \n fit model"
)
axs[0, 2] = _add_loss_histogram(
    axs[0, 2],
    mae_lineshapes,
    title="(c) M3: Fit with reference \n lineshapes",
)
axs[0, 3] = _add_loss_histogram(
    axs[0, 3], mae_nn, title="(D) Convolutional \n Neural network"
)

fig.tight_layout()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\figures"
for ext in [".png", ".eps"]:
    fig_path = os.path.join(save_dir, "hist_fits_single" + ext)
    fig.savefig(fig_path, bbox_inches="tight")

dfs = {
    "Neural network": df_nn,
    "Biesinger": df_biesinger,
    "Fit model": df_fit_model,
    "Lineshapes": df_lineshapes,
}

# %%
threshold = 0.05
indices = (
    df_nn[df_nn["MAE"] < threshold]
    .sort_values(by=["MAE"], ascending=False)
    .index[:10]
)

for index in indices:
    df_print = pd.concat(
        [df_true.loc[index], df_nn.loc[index]], axis=1
    )  # .reset_index()
    df_print["diff"] = (df_true.loc[index] - df_nn.loc[index]) * 100
    print(df_print)
    print(f"Max diff.: {max(df_print['diff'])}")


def print_diff(df, threshold, kind="average"):
    df_test = df.copy()
    correct = 0
    wrong = 0
    for index in df.index:
        df_print = pd.concat(
            [df_true.loc[index], df_test.loc[index]], axis=1
        )  # .reset_index()
        df_print.columns = ["true", "nn"]
        df_print["diff"] = (df_true.loc[index] - df_test.loc[index]) * 100
        # print(f"{index}: max diff.: {max(df_print['diff'])}, mean diff.: {np.mean(np.abs(df_print['diff']))}")
        if kind == "average":
            metric = np.mean(np.abs(df_print["diff"]))
        elif kind == "max":
            metric = np.max(df_print["diff"])
        if metric < threshold:
            correct += 1
        else:
            wrong += 1
    print(
        f"No. of correct classifications ({kind}): {correct}/{len(df_nn.index)} = {correct/len(df_nn.index)*100} %"
    )
    print(
        f"No. of wrong classifications ({kind}): {wrong}/{len(df_nn.index)} = {wrong/len(df_nn.index)*100} %"
    )


threshold = 10.0  # percent
for method, df in dfs.items():
    print(method)
    print_diff(df, threshold, kind="average")
    print_diff(df, threshold, kind="max")
