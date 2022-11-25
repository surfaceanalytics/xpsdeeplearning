# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:37:09 2022

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import string

from sklearn.metrics import mean_absolute_error

os.chdir(
    os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
)

from xpsdeeplearning.network.data_handling import DataHandler

#%%
class Wrapper:
    """Parser for XPS data stored in TXT files."""

    def __init__(self, runfolder):
        """
        Initialize empty data dictionary.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        self.runfolder = runfolder
        self.fontdict = {"size": 55}

        self.results = {}

    def load_predictions(self, clf_name):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------

        """
        pkl_path = os.path.join(
            *[self.runfolder, clf_name, "logs", "results.pkl"]
        )

        with open(pkl_path, "rb") as pickle_file:
            results = pickle.load(
                pickle_file,
            )

        self.results["y_test"] = results["y_test"]
        self.results["pred_test"] = results["pred_test"]

        return self.results["y_test"], self.results["pred_test"]

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

        losses_test = np.array(
            [
                loss_func(
                    self.results["y_test"][i], self.results["pred_test"][i]
                )
                for i in range(self.results["y_test"].shape[0])
            ]
        )
        print(
            f"Median loss: {np.round(np.median(losses_test),2)}",
            f"25th/75th percentile: {np.round(np.percentile(losses_test, [25, 75]),2)}",
        )
        print("Done!")

        self.results["losses_test"] = losses_test

        return losses_test

    def load_sim_values(self, datapath):
        print("Loading data...")
        datahandler = DataHandler(intensity_only=False)

        train_test_split = 0.2
        train_val_split = 0.2
        no_of_examples = 250000

        loaded_data = datahandler.load_data_preprocess(
            input_filepath=datapath,
            no_of_examples=no_of_examples,
            train_test_split=train_test_split,
            train_val_split=train_val_split,
            shuffle=False,
        )

        self.sim_values_test = loaded_data[-1]

        return self.sim_values_test

    def plot_all(self, keys=["noise"]):
        self.x_labels = {
            "shift_x": "Absolute Binding\nEnergy Shift (eV)",
            "noise": "S/N ratio",
            "fwhm": "FWHM of Gaussian\nBroadening Peak (eV)",
            "scatterer": "Scattering Medium",
            "distance": "Distance Travelled\nin Gas Phase (mm)",
            "pressure": "Gas Phase Pressure (mbar)",
        }

        titles = [f"({s})" for s in list(string.ascii_lowercase)]

        ncols = len(keys)

        self.fig, self.axs = plt.subplots(
            1,
            ncols,
            figsize=(16 * ncols, 24),
            dpi=300,
            gridspec_kw={"hspace": 0.4, "wspace": 0.3},
            squeeze=False,
        )

        for i, key in enumerate(keys):
            if key == "fwhsm":
                clf_name = "20220901_13h40m_Mn_linear_combination_normalized_inputs_no_scattering_medium_broadening"
                datapath = r"C:\Users\pielsticker\Simulations\20220901_Mn_linear_combination_small_gas_phase_medium_broadening\20220901_Mn_linear_combination_small_gas_phase_medium_broadening.h5"
                y_test, pred_test = self.load_predictions(clf_name)
                losses_test = self.calculate_test_losses(
                    loss_func=mean_absolute_error
                )
                sim_values_test = self.load_sim_values(datapath)

            sorted_arrays = np.array(
                sorted(
                    [
                        (s[0], l)
                        for (s, l) in zip(
                            self.sim_values_test[key],
                            self.results["losses_test"],
                        )
                    ],
                    reverse=False,
                )
            )

            self.axs[0, i].scatter(
                np.abs(sorted_arrays[:, 0]), sorted_arrays[:, 1]
            )

            self.axs[0, i].set_ylim(0, None)
            self.axs[0, i].set_xlabel(
                self.x_labels[key], fontdict=self.fontdict
            )
            self.axs[0, i].set_ylabel(
                "Mean Absolute Error", fontdict=self.fontdict
            )
            self.axs[0, i].tick_params(
                axis="x", labelsize=self.fontdict["size"]
            )
            self.axs[0, i].tick_params(
                axis="y", labelsize=self.fontdict["size"]
            )
            self.axs[0, i].set_title(titles[i], fontdict=self.fontdict)
            self.axs[0, i].tick_params(axis="x", pad=12)

        self.fig.tight_layout()

        return self.fig, self.axs


#%%
runfolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs"
clf_name = "20220830_09h42m_Mn_linear_combination_normalized_inputs_small_gas_phase_predict_using_20220628_11h57m"
datapath = r"C:\Users\pielsticker\Simulations\20220624_Mn_linear_combination_small_gas_phase\20220624_Mn_linear_combination_small_gas_phase.h5"

wrapper = Wrapper(runfolder)
y_test, pred_test = wrapper.load_predictions(clf_name)
losses_test = wrapper.calculate_test_losses(loss_func=mean_absolute_error)
sim_values_test = wrapper.load_sim_values(datapath)
fig, ax = wrapper.plot_all(keys=["shift_x", "noise", "fwhm"])

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\figures"
fig_filename = "sim_values_effect.png"
fig_path = os.path.join(save_dir, fig_filename)
fig.savefig(fig_path)

# %% See MAE results for individual parameter.
def print_threshold_info(param_name, th_param, ths_mae, reverse=False):
    sorted_arrays = np.array(
        sorted(
            [
                (x[0], i)
                for (x, i) in zip(
                    wrapper.sim_values_test[param_name],
                    wrapper.results["losses_test"],
                )
            ],
            reverse=reverse,
        )
    )
    if reverse:
        comp = ">"
        sim_th = np.where(sorted_arrays[:, 0] > th_param)[0]
    else:
        comp = "<"
        sim_th = np.where(sorted_arrays[:, 0] < th_param)[0]

    no_sim_th = sim_th.shape[0]
    if no_sim_th > 0:
        print(
            f"Of {no_sim_th} test spectra with {param_name} {comp} {th_param},"
        )

        for th_mae in ths_mae:
            if reverse:
                both_th = np.where(sorted_arrays[sim_th, 1] > th_mae)[0]
            else:
                both_th = np.where(sorted_arrays[sim_th, 1] < th_mae)[0]

            no_both_th = both_th.shape[0]
            ratio = np.round((no_both_th / no_sim_th) * 100, 2)
            print(
                f"\t{no_both_th} ({ratio} %) had a "
                f"MAE smaller than {th_mae}."
            )
    else:
        print(f"No test spectrum had {param_name} {comp} {th_param}.")


ths_noise = [5, 10, 20, 35]
ths_mae = [0.05, 0.1, 0.15, 0.2]

for th_noise in ths_noise:
    print_threshold_info(
        param_name="noise", th_param=th_noise, ths_mae=ths_mae, reverse=False
    )
    print("\n")

#%%
import pandas as pd
import seaborn as sns

d_sim_values = {
    "shift_x": np.abs(wrapper.sim_values_test["shift_x"][:, 0]),
    "noise": wrapper.sim_values_test["noise"][:, 0],
    "fwhm": wrapper.sim_values_test["fwhm"][:, 0],
    # "scatterer": wrapper.sim_values_test["scatterer"][:,0],
    # "distance": wrapper.sim_values_test["distance"][:,0],
    # "pressure": wrapper.sim_values_test["pressure"][:,0],
}
df_out = pd.DataFrame(d_sim_values)
df_out.insert(0, "MAE", wrapper.results["losses_test"])

corr_data = df_out.corr(method="pearson")
mask = np.identity(4)

fig, ax = plt.subplots(figsize=(2, 4))
p = sns.heatmap(
    corr_data[["MAE"]],
    ax=ax,
    linewidths=0.1,
    linecolor="white",
    annot=True,
    cmap="YlOrRd",
)
ax.set_title("Correlation of the scanned parameters")

p.set_xticklabels(corr_data[["MAE"]], rotation=0)
p.set_yticklabels(corr_data[["MAE"]].index, rotation=0)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(13)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(13)

#%%
# =============================================================================
# clf_name = "20220831_10h44m_Mn_linear_combination_normalized_inputs_no_scattering_strong_broadening"
# datapath = r"C:\Users\pielsticker\Simulations\20220830_Mn_linear_combination_small_gas_phase_strong_broadening\20220830_Mn_linear_combination_small_gas_phase_strong_broadening.h5"
# y_test, pred_test = wrapper.load_predictions(clf_name)
# losses_test = wrapper.calculate_test_losses(loss_func=mean_absolute_error)
# sim_values_test = wrapper.load_sim_values(datapath)
# fig, ax = wrapper.plot_all(keys=["shift_x", "noise", "fwhm"])
#
# sorted_arrays = np.array(sorted(
#     [
#         (s[0], l)
#         for (s, l) in zip(wrapper.sim_values_test["fwhm"],
#                           wrapper.results["losses_test"])
#         ],
#     reverse=False
#     )
# )
#
# low_fwhm_indices = np.where(sorted_arrays[:,0] < 1.0)
# high_fwhm_indices = np.where(sorted_arrays[:,0] > 7.5)
#
# losses_low_fwhm = np.median(sorted_arrays[low_fwhm_indices][:,1])
# losses_high_fwhm = np.median(sorted_arrays[high_fwhm_indices][:,1])
#
# =============================================================================
