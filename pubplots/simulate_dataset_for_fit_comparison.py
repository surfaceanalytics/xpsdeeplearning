# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:57:39 2022

@author: pielsticker
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:22:27 2022

@author: pielsticker
"""

import os
import json
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(
    os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
)

from xpsdeeplearning.simulation.base_model.spectra import MeasuredSpectrum
from xpsdeeplearning.simulation.creator import Creator, FileWriter
from xpsdeeplearning.simulation.base_model.figures import Figure

#%% Input parameter
# Change the following line according to your folder structure ###
init_param_folder = r"C:\Users\pielsticker\Simulations\paper"
# init_param_filename = "init_params_Fe_core_small_gas_phase.json"
init_param_filename = "init_params_NiCoFe_combined_core_small_gas_phase.json"

#%% Parameters
init_param_filepath = os.path.join(init_param_folder, init_param_filename)

with open(init_param_filepath, "r") as param_file:
    params = json.load(param_file)
    params["init_param_filepath"] = init_param_filepath
    params["no_of_simulations"] = 100
    params["scatter"] = False
    params["shift_x"] = False

#%% Create multiple sets of similar spectra with the same settings
creator = Creator(params)
df = creator.run()
creator.plot_random(5)

writer = FileWriter(creator.df, creator.params)

writer.to_file(filetypes=["hdf5"], metadata=True)
with h5py.File(writer.hdf5_filepath, "r") as hf:
    X_h5 = hf["X"][:, :, :]
    y_h5 = hf["y"][:, :]

#%% Save data and metadata.
output_datafolder = os.path.join(
    *[os.getcwd(), "utils", creator.params["name"]]
)
output_datafolder = r"C:\Users\pielsticker\Downloads"

# filepath_output = os.path.join(output_datafolder, "truth.csv")
filepath_output = os.path.join(output_datafolder, "truth_multiple.csv")

indices = [
    f"Synthetic spectrum no. {str(i).zfill(len(str(creator.no_of_simulations-1)))}"
    for i in range(creator.no_of_simulations)
]
df_true = pd.DataFrame(y_h5, index=indices, columns=creator.labels)
df_true.to_csv(filepath_output, index=True, sep=";")

# %%
filename = "ones.vms"
# input_datafolder = os.path.join(os.getcwd(), "utils")
input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
filepath_input = os.path.join(input_datafolder, filename)
spectrum = MeasuredSpectrum(filepath_input)

for i in range(creator.no_of_simulations):
    row = creator.df.iloc[i]
    x = row["x"]
    y = row["y"]
    title = "Simulated spectrum no." + str(i)
    fig = Figure(x, y, title)
    spectrum_text = creator._write_spectrum_text(row)
    fig.ax.text(
        0.1,
        0.9,
        spectrum_text,
        horizontalalignment="left",
        verticalalignment="top",
        transform=fig.ax.transAxes,
        fontsize=7,
    )
    plt.show()

    spectrum.x = df["x"][i]
    spectrum.lineshape = np.squeeze(df["y"][i])
    spectrum.label = df["label"][i]
    number = str(i).zfill(len(str(creator.no_of_simulations - 1)))
    spectrum.converter.data[0][
        "group_name"
    ] = f"Synthetic spectrum no. {number}"
    spectrum.converter.data[0]["settings"]["nr_values"] = spectrum.x.shape[0]
    new_filename = f"Synthetic spectrum {number}.vms"

    datafolder_vms = os.path.join(output_datafolder, "multiple")
    spectrum.write(datafolder_vms, new_filename)

    # spectrum.write(output_datafolder, new_filename)
    print(spectrum.label)