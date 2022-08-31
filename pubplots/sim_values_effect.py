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
from common import maximum_absolute_error

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
        self.fontdict = {"size": 42}
        self.fontdict_small = {"size": 14}
        self.fontdict_legend = {"size": 16}

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
                loss_func(self.results["y_test"][i], self.results["pred_test"][i])
                for i in range(self.results["y_test"].shape[0])
            ]
        )
        print(
            np.median(losses_test),
            np.percentile(losses_test, [25, 75]),
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

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            self.sim_values_test,
        ) = datahandler.load_data_preprocess(
            input_filepath=datapath,
            no_of_examples=no_of_examples,
            train_test_split=train_test_split,
            train_val_split=train_val_split,
            shuffle=False,
        )

        return self.sim_values_test


    def plot_all(self, keys=["noise"]):
        self.x_labels = {
            "shift_x": "Absolute shift along the BE axis",
            "noise": "S/N ratio",
            "fwhm": "FWHM of Gaussian broadening peak",
            "scatterer": "Scattering medium",
            "distance": "Distance travelled in gas phase",
            "pressure": "Gas phase pressure",
            }

        titles = [f"({s})" for s in list(string.ascii_lowercase)]

        ncols = len(keys)

        self.fig, self.axs = plt.subplots(
            1,
            ncols,
            figsize=(16*ncols, 24),
            dpi=300,
            squeeze=False
            )

        for i, key in enumerate(keys):
            sorted_arrays = np.array(sorted(
                [
                    (x[0], i)
                    for (x, i) in zip(self.sim_values_test[key],
                                      self.results["losses_test"])
                    ],
                reverse=False
                )
            )

            if key == "shift_x":
                self.axs[0,i].scatter(
                    np.abs(sorted_arrays[:,0]),
                    sorted_arrays[:,1])
            else:#if key == "shift_x":
                self.axs[0,i].scatter(
                    np.abs(sorted_arrays[:,0]),
                    sorted_arrays[:,1])

            self.axs[0,i].set_xlabel(key.capitalize(), fontdict=self.fontdict)
            self.axs[0,i].set_ylabel("Mean Absolute Error", fontdict=self.fontdict)
            self.axs[0,i].tick_params(axis="x", labelsize=self.fontdict["size"])
            self.axs[0,i].tick_params(axis="y", labelsize=self.fontdict["size"])
            self.axs[0,i].set_title(titles[i], fontdict=self.fontdict)

        self.fig.tight_layout()

        return self.fig, self.axs


#%%
runfolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs"
clf_name = "20220830_16h30m_Mn_linear_combination_normalized_inputs_small_gas_phase_strong_broadening_using_20220628_11h57m"
datapath = r"C:\Users\pielsticker\Simulations\20220830_Mn_linear_combination_small_gas_phase_strong_broadening\20220830_Mn_linear_combination_small_gas_phase_strong_broadening.h5"

wrapper = Wrapper(runfolder)
y_test, pred_test = wrapper.load_predictions(clf_name)
losses_test = wrapper.calculate_test_losses(loss_func=mean_absolute_error)
sim_values_test = wrapper.load_sim_values(datapath)
fig, ax = wrapper.plot_all(keys=["shift_x", "noise", "fwhm"])
plt.show()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures"
fig_filename = "sim_values_effect.png"
fig_path = os.path.join(save_dir, fig_filename)
fig.savefig(fig_path)

# %% See results for noise.
sorted_arrays = np.array(sorted(
    [
        (x[0], i)
        for (x, i) in zip(wrapper.sim_values_test["noise"],
                          wrapper.results["losses_test"])
        ],
    reverse=False
    )
)

np.where(sorted_arrays[np.where(sorted_arrays[:,1] < 0.1)[0], 0] < 5)[0].shape
np.where(sorted_arrays[:,0] < 5)[0].shape

#%%
import pandas as pd
import seaborn as sns

d_sim_values = {
    "shift_x": np.abs(wrapper.sim_values_test["shift_x"][:,0]),
    "noise": wrapper.sim_values_test["noise"][:,0],
    "fwhm": wrapper.sim_values_test["fwhm"][:,0],
    #"scatterer": wrapper.sim_values_test["scatterer"][:,0],
    #"distance": wrapper.sim_values_test["distance"][:,0],
    #"pressure": wrapper.sim_values_test["pressure"][:,0],
    }
df_out = pd.DataFrame(d_sim_values)
df_out.insert(0, "MAE", wrapper.results["losses_test"])

corr_data = df_out.corr(method="pearson")

fig, ax = plt.subplots()
p = sns.heatmap(
    corr_data,
    ax=ax,
    linewidths=0.1,
    linecolor="white",
    cmap="YlOrRd",
)
ax.set_title(
    "Correlation matrix of the scanned parameters",
    #wrapper.fontdict,
)
p.set_xticklabels(corr_data, rotation=90)
p.set_yticklabels(corr_data, rotation=0)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(13)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(13)