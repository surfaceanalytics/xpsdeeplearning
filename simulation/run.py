# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020.

@author: pielsticker

This script is used to simulate many spectra using the Creator class
and save the data to a HDF 5 file, along with the metadata in a JSON
file.
github
"""

from time import time
import os
import json
import h5py
import pandas as pd

from creator import Creator, FileWriter, calculate_runtime

#%% Input parameter
# Change the following line according to your folder structure ###
init_param_folder = r"C:\Users\pielsticker\Simulations\params_CoFe"
init_param_filename = "init_params_CoFe_combined_core_auger_100eV_window.json"
#%% Parameters
init_param_filepath = os.path.join(init_param_folder, init_param_filename)

with open(init_param_filepath, "r") as param_file:
    params = json.load(param_file)
    params["init_param_filepath"] = init_param_filepath

#%% Create multiple sets of similar spectra with the same settings
runtimes = {}

t0 = time()
creator = Creator(params)
df = creator.run()
creator.plot_random(10)
t1 = time()
runtime = calculate_runtime(t0, t1)
runtimes["dataset_creation"] = runtime
print(f"Runtime: {runtime}.")

#%% Save data and metadata.
t0 = time()
writer = FileWriter(creator.df, creator.params)
writer.to_file(filetypes=["pickle"], metadata=False)
t1 = time()
runtimes["pkl_save"] = calculate_runtime(t0, t1)
print(f"Pickle save runtime: {runtimes['pkl_save']}.")

#%% Save data and metadata.
t0 = time()

writer.to_file(filetypes=["hdf5"], metadata=True)
t1 = time()
runtimes["h5_save"] = calculate_runtime(t0, t1)
print(f"HDF5 save runtime: {runtimes['h5_save']}.")

# Test new file.
t0 = time()
with h5py.File(writer.hdf5_filepath, "r") as hf:
    size = hf["X"].shape
    X_h5 = hf["X"][:4000, :, :]
    y_h5 = hf["y"][:4000, :]
    shiftx_h5 = hf["shiftx"][:4000, :]
    noise_h5 = hf["noise"][:4000, :]
    fwhm_h5 = hf["FWHM"][:4000, :]
    scatterers_h5 = hf["scatterer"][:4000, :]
    energies_h5 = hf["energies"][:4000]
    labels_h5 = [str(label) for label in hf["labels"]]

t1 = time()
runtimes["h5_load"] = calculate_runtime(t0, t1)
print(f"HDF5 load runtime: {runtimes['h5_load']}.")

if creator.params["eV_window"]:
    creator.plot_random(10)

os.remove(writer.pkl_filepath)

#%% Reload pickle file if needed.
# =============================================================================
# t0 = time()
# creator.df = pd.read_pickle(writer.pkl_filepath)
# t1 = time()
# runtimes["pkl_load"] = calculate_runtime(t0, t1)
# print(f"Pickle load runtime: {runtimes['pkl_load']}.")
# 
# =============================================================================
#%% If run was interrupted
# =============================================================================
# runtimes = {}
# datafolder = r"C:\Users\pielsticker\Simulations"
# dataset_name = "20210910_FeCo_combined_without_auger_peaks_100_eV_window"
# 
# # Reload pickle file.
# pkl_filename = dataset_name + ".pkl"
# pkl_filepath = os.path.join(*[datafolder, dataset_name, pkl_filename])
# 
# t0 = time()
# df = pd.read_pickle(pkl_filepath)
# t1 = time()
# runtimes["pkl_load"] = calculate_runtime(t0, t1)
# print(f"Pickle load runtime: {runtimes['pkl_load']}.")
# 
# # Reload parameters .
# param_filename = dataset_name + "_metadata.json"
# param_filepath = os.path.join(*[datafolder, dataset_name, param_filename])
# 
# with open(param_filepath, "r") as param_file:
#     params = json.load(param_file)
# 
# writer = FileWriter(df, params)
# writer.to_file(filetypes=["hdf5"], metadata=True)
# t1 = time()
# runtimes["h5_save"] = calculate_runtime(t0, t1)
# print(f"HDF5 save runtime: {runtimes['h5_save']}.")
# 
# os.remove(pkl_filepath)
# =============================================================================