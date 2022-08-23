# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:06:37 2020.

@author: pielsticker


This script can be used to load reference and fitted XPS spectra and
resize them according to new start, stop, and step values.

To do this, the data are stored in ReferenceSpectrum or FittedSpectrum
instances, respectively. Then all points outside the interval
(start, stop) are removed and, if needed, the lineshape is extended to
reach the start and stop values Finally, the x and lineshape values are
up-/downsampled according to the new step value.

For fitted spectra, it is possible to load the data from multiple .xy
files and store them in a hdf5 file as a test data set for the neural
network studies.
"""

import numpy as np
import os
import h5py
import pandas as pd

from base_model.spectra import (
    Spectrum,
    MeasuredSpectrum,
    FittedSpectrum,
    SimulatedSpectrum,
)
from base_model.figures import Figure

#%% For one reference spectrum.
input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
filename = "NiCoFe.vms"

energies = []
filepath = os.path.join(input_datafolder, filename)
ref_spectrum = MeasuredSpectrum(filepath)
fig_old = Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="old")
energies.append(ref_spectrum.x[np.argmax(ref_spectrum.lineshape)])

ref_spectrum.resample(start=682.0, stop=885.0, step=0.2)
energies.append(ref_spectrum.x[np.argmax(ref_spectrum.lineshape)])
fig_new = Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="new")
new_filename = filename.split(".")[0] + "_new.vms"
ref_spectrum.write(input_datafolder, new_filename)

# =============================================================================
# filename_np = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\nicofe_mixed"
#
# np.save(filename_np,
#            ref_spectrum.lineshape)
# load_ls = np.load(filename_np + ".npy")
#
# fig_new = Figure(x=ref_spectrum.x, y=load_ls, title="load")
# =============================================================================

# =============================================================================
# #%% To convert TXT to VAMAS
# input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
# filename0 = "nicofe.vms"
# input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\xpsdeeplearning\data\references"
# filename1 = "Pd3d_PdO_narrow.txt"
#
# filepath0 = os.path.join(input_datafolder, filename0)
# ref_spectrum0 = MeasuredSpectrum(filepath0)
# filepath1 = os.path.join(input_datafolder, filename1)
# ref_spectrum1 = MeasuredSpectrum(filepath1)
#
# fig_old = Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="old")
#
# #d = ref_spectrum0.converter.data[0]
# ref_spectrum0.converter.data[0]["group_name"] = "PdO"
# ref_spectrum0.converter.data[0]["settings"]["nr_values"] = ref_spectrum1.lineshape.shape[0]
# ref_spectrum0.x = ref_spectrum1.x
# ref_spectrum.lineshape = ref_spectrum1.lineshape
#
# ref_spectrum.converter.data[0]["spectrum_type"] = "Pd 3d"
# ref_spectrum.converter.data[0]["data"]["x"] = list(ref_spectrum1.x)
# ref_spectrum.converter.data[0]["data"]["y0"] = list(ref_spectrum1.lineshape)
# ref_spectrum.converter.data[0]["data"]["y1"] = ref_spectrum0.converter.data[0]["data"]["y1"][:ref_spectrum1.lineshape.shape[0]]
#
# fig_new = Figure(x=ref_spectrum1.x, y=ref_spectrum1.lineshape, title="new")
# new_filename = filename1 + "_new.vms"
# ref_spectrum.write(input_datafolder, new_filename)
#
# =============================================================================
#%% For one fitted XPS spectrum
# =============================================================================
# input_datafolder = r'C:\Users\pielsticker\Desktop\Mixed Fe spectra\exported'
# filename = 'measured0001.txt'
# energies = []
#
# filepath = os.path.join(input_datafolder, filename)
# fit_spectrum = FittedSpectrum(filepath)
# fig_old = Figure(x=fit_spectrum.x,
#                  y=fit_spectrum.lineshape,
#                  title='old')
# energies.append(fit_spectrum.x[np.argmax(fit_spectrum.lineshape)])
#
# fit_spectrum.resample(start=694, stop=750, step=0.05)
# fig_new = Figure(x=fit_spectrum.x,
#                  y=fit_spectrum.lineshape,
#                  title='new')
# energies.append(fit_spectrum.x[np.argmax(fit_spectrum.lineshape)])
# # fit_spectrum.write(input_datafolder)
# =============================================================================

#%% For multiple fitted XPS spectra
def _get_labels(filepath, label_list):
    """
    Take the labels and names from the excel file in the filepath.

    Parameters
    ----------
    filepath : str
        Filepath of the excel file, has to be .xlsx.
        labels: list
    label_list: list
        A list of labels that needs to match the column names in the
        Excel file.

    Returns
    -------
    y : ndarray
        2D array of labels.
    names : ndarray
        Names of the spectra.

    """
    df = pd.read_excel(filepath, engine="openpyxl")

    columns = []
    for key in label_list:
        columns.append(df[key])
    y = np.transpose(np.array(columns))

    names = np.reshape(np.array(list(df["name"])), (-1, 1))

    return y, names


def convert_all_spectra(
    input_datafolder, label_filepath, label_list, plot_all=True
):
    """
    Take all xy files of measured spectra in the input_datafolder.

    Features, labels, and names are extracted. If neeeded, the spectra
    are resized.

    Parameters
    ----------
    input_datafolder : str
        Folder of the exported XPS spectra.
    label_filepath : str
        Filepath of the excel file, has to be .xlsx.
    label_list: list
        A list of labels that needs to match the column names in the
        Excel file.
    plot_all : bool, optional
        If plot_all, all loadded spectra are plotted.
        The default is True.

    Returns
    -------
    X : ndarray
        3D array of XPS data.
    y : ndarray
        2D array of labels.
    names : ndarray
        Spectra names.
    energies: ndarray
        1D array of binding energies
    """
    import warnings

    warnings.filterwarnings("ignore")
    filenames = next(os.walk(input_datafolder))[2]

    X = np.zeros((len(filenames), 381, 1))

    # y, names = _get_labels(label_filepath, label_list)
    spectra = []

    for name in filenames:
        filepath = os.path.join(input_datafolder, name)
        spectrum = FittedSpectrum(filepath)
        index = filenames.index(name)
        spectrum.resample(start=331, stop=350, step=0.05)
        spectrum.normalize()
        spectra.append(spectrum)
        X[index] = np.reshape(spectrum.lineshape, (-1, 1))

        if plot_all:
            text = (
                "Spectrum no. "
                + str(spectrum.number)
                + "\n"
                + str(spectrum.label)
            )
            Figure(spectrum.x, spectrum.lineshape, title=text)

    energies = spectrum.x

    # return X, y, names, energies
    return X, names, energies


# Load the data into numpy arrays and save to hdf5 file.
input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Data\NAP-XPS\analyzed data\Pd spectra\exported"
label_filepath = r"C:\Users\pielsticker\Lukas\MPI-CEC\Data\NAP-XPS\analyzed data\Pd spectra\analysis_20210122\peak fits_20210122.xlsx"
label_list = ["Pd metal", "PdO"]
X, y, names, energies = convert_all_spectra(
    input_datafolder, label_filepath, label_list, plot_all=False
)
output_file = (
    r"C:\Users\pielsticker\Simulations\20210125_palladium_measured_caro_fit.h5"
)

with h5py.File(output_file, "w") as hf:
    hf.create_dataset(
        "X",
        data=X,
        compression="gzip",
        chunks=True,
        maxshape=(None, X.shape[1], X.shape[2]),
    )
    hf.create_dataset(
        "y",
        data=y,
        compression="gzip",
        chunks=True,
        maxshape=(None, y.shape[1]),
    )
    hf.create_dataset(
        "names",
        data=np.array(names, dtype="S"),
        compression="gzip",
        chunks=True,
        maxshape=(None, names.shape[1]),
    )
    hf.create_dataset(
        "energies", data=energies, compression="gzip", chunks=True
    )
    labels = np.array(label_list, dtype=object)
    string_dt = h5py.special_dtype(vlen=str)
    hf.create_dataset(
        "labels",
        data=labels,
        dtype=string_dt,
        compression="gzip",
        chunks=True,
    )

# Test the new file
with h5py.File(output_file, "r") as hf:
    size = hf["X"].shape
    X_h5 = hf["X"][:, :, :]
    y_h5 = hf["y"][:, :]
    names_h5 = hf["names"][:]
    energies_h5 = hf["energies"][:]
    labels_h5 = [str(label) for label in hf["labels"][:]]

#%% For converting spectra where the composition is unknown.
def load_data(filepath):
    """
    Overwrite load method from the MeasuredSpectrum class.

    This is done to accomodate header of CasaXPS export and
    associate the spectrum with a number.

    Parameters
    ----------
    filepath: str
        Filepath of the .txt file with the data in the format of
        x and y values.

    Returns
    -------
    label : dict
        Name of the spectrum given in the first row of the xy file.
    data : ndarray
        2D array with the x and y values from the file.

    """
    lines = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            lines += [line]
    # This takes the species given in the first line
    name = lines[0].split(":")[1]
    species = str(lines[0]).split(":")[-1].split("\n")[0]
    lines = [[float(i) for i in line.split()] for line in lines[8:]]
    xy_data = np.array(lines)[:, 2:]

    data = {
        "data": {
            "x": list(xy_data[:, 0]),
            "y0": list(xy_data[:, 1]),
        },
        "spectrum_type": species,
        "name": name,
    }
    return data


def convert_all_spectra(input_datafolder, plot_all=True):
    """
    Take all xy files of measured spectra in the input_datafolder.

    Features, labels, and names are extracted. If neeeded, the spectra
    are resized.

    Parameters
    ----------
    input_datafolder : str
        Folder of the exported XPS spectra.
    plot_all : bool, optional
        If plot_all, all loadded spectra are plotted.
        The default is True.

    Returns
    -------
    X : ndarray
        3D array of XPS data.
    y : ndarray
        2D array of labels.
    names : ndarray
        Spectra names.
    energies: ndarray
        1D array of binding energies
    """
    import warnings

    warnings.filterwarnings("ignore")
    # filenames = next(os.walk(input_datafolder))[2]
    filenames = ["nicofe_mixed.txt"]

    X = np.zeros((len(filenames), 1901, 1))

    spectra = []
    names = []

    for filename in filenames:
        filepath = os.path.join(input_datafolder, filename)
        data = load_data(filepath)

        x = np.array(data["data"]["x"])
        # x -= 0.808
        x1 = np.roll(x, -1)
        diff = np.abs(np.subtract(x, x1))
        step = np.round(np.min(diff[diff != 0]), 3)
        x = x[diff != 0]
        y = np.array(data["data"]["y0"])[diff != 0]

        start = np.round(np.min(x), 3)
        stop = np.round(np.max(x), 3)
        name = data["name"]
        names.append(name)

        spectrum = Spectrum(start, stop, step, name)
        spectrum.x = np.round(x, 3)
        spectrum.lineshape = y

        index = filenames.index(filename)

        spectrum.resample(start=700.0, stop=890.0, step=0.1)
        spectrum.normalize()
        spectra.append(spectrum)
        X[index] = np.reshape(spectrum.lineshape, (-1, 1))

        if plot_all:
            Figure(spectrum.x, spectrum.lineshape, title=spectrum.label)

        energies = spectrum.x

    return X, names, energies


# Load the data into numpy arrays and save to hdf5 file.
# =============================================================================
# input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Data\NAP-XPS\analyzed data\AmmoMaxRef\exports"
# X, names, energies = convert_all_spectra(input_datafolder, plot_all=True)
# output_file = r"C:\Users\pielsticker\Simulations\20220609_AmmoMax_spectra.h5"
# =============================================================================
input_datafolder = (
    r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
)
X, names, energies = convert_all_spectra(input_datafolder, plot_all=True)
output_file = r"C:\Users\pielsticker\Simulations\20220721_NiCoFe.h5"

y = np.expand_dims(
    np.array([0.018, 0.234, 0.024, 0.035, 0.367, 0.001, 0.032, 0.001, 0.288]),
    axis=0,
)

X = np.repeat(X, 2, axis=0)
y = np.repeat(y, 2, axis=0)
names = names + names

label_list = [
    "Ni2pCo2pFe2p Ni metal",
    "Ni2pCo2pFe2p NiO",
    "Ni2pCo2pFe2p Co metal",
    "Ni2pCo2pFe2p CoO",
    "Ni2pCo2pFe2p Co3O4",
    "Ni2pCo2pFe2p Fe metal",
    "Ni2pCo2pFe2p FeO",
    "Ni2pCo2pFe2p Fe3O4",
    "Ni2pCo2pFe2p Fe2O3",
]

with h5py.File(output_file, "w") as hf:
    hf.create_dataset(
        "X",
        data=X,
        compression="gzip",
        chunks=True,
        maxshape=(None, X.shape[1], X.shape[2]),
    )
    hf.create_dataset(
        "y",
        data=y,
        compression="gzip",
        chunks=True,
        maxshape=(None, y.shape[1]),
    )
    hf.create_dataset(
        "energies", data=energies, compression="gzip", chunks=True
    )
    string_dt = h5py.special_dtype(vlen=str)

    labels = np.array(label_list, dtype=object)
    string_dt = h5py.special_dtype(vlen=str)
    hf.create_dataset(
        "labels",
        data=labels,
        dtype=string_dt,
        compression="gzip",
        chunks=True,
    )

# =============================================================================
#     hf.create_dataset(
#         "names",
#         data=names,
#         dtype=string_dt,
#         compression="gzip",
#         chunks=True,
#     )
# =============================================================================

# Test the new file
with h5py.File(output_file, "r") as hf:
    size = hf["X"].shape
    X_h5 = hf["X"][:, :, :]
    y_h5 = hf["y"][:, :]
    energies_h5 = hf["energies"][:]
    # names_h5 = [str(name.decode("utf-8")) for name in hf["names"][:]]
    labels_h5 = [str(label) for label in hf["labels"][:]]
