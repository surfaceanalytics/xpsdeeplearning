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
This script can be used to load reference and fitted XPS spectra and
resize them according to new start, stop, and step values.

To do this, the data are stored in ReferenceSpectrum or FittedSpectrum
instances, respectively. Then all points outside the interval
(start, stop) are removed and, if needed, the lineshape is extended to
reach the start and stop values. Finally, the x and lineshape values are
up-/downsampled according to the new step value.

For fitted spectra, it is possible to load the data from multiple .txt
files and store them in a hdf5 file as a test data set for the neural
network studies.
"""
import os
from typing import List
import warnings
import h5py
import numpy as np
import pandas as pd

from xpsdeeplearning.simulation.base_model.figures import Figure
from xpsdeeplearning.simulation.base_model.spectra import (
    FittedSpectrum,
    MeasuredSpectrum,
    Spectrum,
)


def resample_one_reference_spectrum(
    input_datafolder: str,
    filename: str,
    new_start: float,
    new_stop: float,
    new_step: float,
    save_to_npy: bool = True,
):
    """
    Convert one vamas spectrum, resample, and save back to new file.

    Parameters
    ----------
    input_datafolder: str
        Path of the data.
    filename : str
        Data filename.
    new_start : float
        New minimum value of x array.
    new_stop : float
        New maximum value of x array.
    new_step : float
        New step size of x array.
    save_to_npy : bool, optional
        If True, the new spectrum is saved as a .npy file.
        The default is True.

    """
    filepath = os.path.join(input_datafolder, filename)
    ref_spectrum = MeasuredSpectrum(filepath)
    Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="old")

    ref_spectrum.resample(start=new_start, stop=new_stop, step=new_step)
    Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="new")

    new_filename = filename.split(".")[0] + "_new.vms"
    ref_spectrum.write(output_folder=input_datafolder, new_filename=new_filename)

    if save_to_npy:
        filename_np = filepath + ".npy"

        np.save(filename_np, ref_spectrum.lineshape)
        load_ls = np.load(filename_np + ".npy")

        Figure(x=ref_spectrum.x, y=load_ls, title="load")


def resample_one_fitted_spectrum(
    input_datafolder: str,
    filename: str,
    new_start: float,
    new_stop: float,
    new_step: float,
):
    """
    Convert one txt spectrum, resample, and save back to new file.

    Parameters
    ----------
    input_datafolder: str
        Path of the data.
    filename : str
        Data filename.
    new_start : float
        New minimum value of x array.
    new_stop : float
        New maximum value of x array.
    new_step : float
        New step size of x array.

    """
    filepath = os.path.join(input_datafolder, filename)
    fit_spectrum = FittedSpectrum(filepath)
    Figure(x=fit_spectrum.x, y=fit_spectrum.lineshape, title="old")
    fit_spectrum.resample(start=new_start, stop=new_stop, step=new_step)
    Figure(x=fit_spectrum.x, y=fit_spectrum.lineshape, title="new")
    new_filename = filename.split(".")[0] + "_new.vms"
    fit_spectrum.write(output_folder=input_datafolder, new_filename=new_filename)


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
    labels : ndarray
        2D array of labels.
    names : ndarray
        Names of the spectra.

    """
    df = pd.read_excel(filepath, engine="openpyxl")

    columns = []
    for key in label_list:
        columns.append(df[key])
    labels = np.transpose(np.array(columns))

    names = np.reshape(np.array(list(df["name"])), (-1, 1))

    return labels, names


def convert_and_resample_all_exported_spectra(
    input_datafolder: str,
    new_start: float,
    new_stop: float,
    new_step: float,
    plot_all: bool = True,
):
    """
    Take all xy files of measured spectra in the input_datafolder.

    Parameters
    ----------
    input_datafolder : str
        Folder of the exported XPS spectra.
    new_start : float
        New minimum value of x array.
    new_stop : float
        New maximum value of x array.
    new_step : float
        New step size of x array.
    plot_all : bool, optional
        If plot_all, all loadded spectra are plotted.
        The default is True.

    Returns
    -------
    X : ndarray
        3D array of XPS data.
    energies: ndarray
        1D array of binding energies

    """

    warnings.filterwarnings("ignore")
    filenames = next(os.walk(input_datafolder))[2]

    X = np.zeros((len(filenames), 381, 1))

    spectra = []

    for name in filenames:
        filepath = os.path.join(input_datafolder, name)
        spectrum = FittedSpectrum(filepath)
        index = filenames.index(name)
        spectrum.resample(start=new_start, stop=new_stop, step=new_step)
        spectrum.normalize()
        spectra.append(spectrum)
        X[index] = np.reshape(spectrum.lineshape, (-1, 1))

        if plot_all:
            text = "Spectrum no. " + str(spectrum.number) + "\n" + str(spectrum.label)
            Figure(spectrum.x, spectrum.lineshape, title=text)

    energies = spectrum.x

    return X, energies


def convert_multiple_exported_spectra(
    input_datafolder: str,
    new_start: float,
    new_stop: float,
    new_step: float,
    label_filepath: str,
    output_file: str,
    label_list: List = None,
    plot_all: bool = True,
):
    """
    Load the data of multiple exported spectra into numpy arrays
    and save to hdf5 file.
    """
    labels, names = _get_labels(label_filepath, label_list)
    X, y, names, energies = convert_and_resample_all_exported_spectra(
        input_datafolder=input_datafolder,
        new_start=new_start,
        new_stop=new_stop,
        new_step=new_step,
        plot_all=plot_all,
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
        hf.create_dataset("energies", data=energies, compression="gzip", chunks=True)
        labels = np.array(label_list, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        hf.create_dataset(
            "labels",
            data=labels,
            dtype=string_dt,
            compression="gzip",
            chunks=True,
        )


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


def convert_and_resample_all_measured_spectra(
    input_datafolder: str,
    new_start: float,
    new_stop: float,
    new_step: float,
    plot_all: bool = True,
):
    """
    Take all xy files of measured spectra in the input_datafolder.

    Features, labels, and names are extracted. If neeeded, the spectra
    are resized.

    Parameters
    ----------
    input_datafolder : str
        Folder of the exported XPS spectra.
    new_start : float
        New minimum value of x array.
    new_stop : float
        New maximum value of x array.
    new_step : float
        New step size of x array.
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

    X = np.zeros((len(filenames), 1901, 1))

    spectra = []
    names = []

    for filename in filenames:
        filepath = os.path.join(input_datafolder, filename)
        data = load_data(filepath)

        x = np.array(data["data"]["x"])
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

        spectrum.resample(start=new_start, stop=new_stop, step=new_step)
        spectrum.normalize()
        spectra.append(spectrum)
        X[index] = np.reshape(spectrum.lineshape, (-1, 1))

        if plot_all:
            Figure(spectrum.x, spectrum.lineshape, title=spectrum.label)

        energies = spectrum.x

    return X, names, energies


def convert_multiple_measured_spectra(
    input_datafolder: str,
    new_start: float,
    new_stop: float,
    new_step: float,
    output_file: str,
    label_list: List = None,
    plot_all: bool = True,
):
    """
    Convert and saves multiple measured spectra with unknown composition.
    """
    X, names, energies = convert_and_resample_all_measured_spectra(
        input_datafolder=input_datafolder,
        new_start=new_start,
        new_stop=new_stop,
        new_step=new_step,
        plot_all=plot_all,
    )

    with h5py.File(output_file, "w") as hf:
        hf.create_dataset(
            "X",
            data=X,
            compression="gzip",
            chunks=True,
            maxshape=(None, X.shape[1], X.shape[2]),
        )

        hf.create_dataset("energies", data=energies, compression="gzip", chunks=True)
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
        hf.create_dataset(
            "names",
            data=names,
            dtype=string_dt,
            compression="gzip",
            chunks=True,
        )


if __name__ == "__main__":
    """Select one of the converter function below."""

    """For one reference spectrum."""
    # input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
    # filename = "NiCoFe.vms"
    # new_start, new_stop, new_step = 682.0, 885.0, 0.2
    # resample_one_reference_spectrum(
    #     input_datafolder=input_datafolder,
    #     filename=filename,
    #     new_start=new_start,
    #     new_stop=new_stop,
    #     new_step=new_step,
    #     save_to_npy=False
    #     )

    """For one fitted XPS spectrum."""
    # input_datafolder = r"C:\Users\pielsticker\Desktop\Mixed Fe spectra\exported"
    # filename = "measured0001.txt"
    # new_start, new_stop, new_step = 694.0, 750.0, 0.05
    # resample_one_fitted_spectrum(
    #     input_datafolder=input_datafolder,
    #     filename=filename,
    #     new_start=new_start,
    #     new_stop=new_stop,
    #     new_step=new_step,
    #     save_to_npy=False
    #     )

    """For multiple fitted XPS spectra."""
    # input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Data\NAP-XPS\analyzed data\Pd spectra\exported"
    # label_filepath = r"C:\Users\pielsticker\Lukas\MPI-CEC\Data\NAP-XPS\analyzed data\Pd spectra\analysis_20210122\peak fits_20210122.xlsx"
    # output_file = r"C:\Users\pielsticker\Simulations\20210125_palladium_measured_caro_fit.h5"

    # label_list = ["Pd metal", "PdO"]
    # new_start, new_stop, new_step = 331, 350, 0.05

    # convert_multiple_exported_spectra(
    #         input_datafolder=input_datafolder,
    #         new_start=new_start,
    #         new_stop=new_stop,
    #         new_step=new_step,
    #         label_filepath=label_filepath,
    #         label_list=label_list,
    #         plot_all=True,
    # )

    """For converting spectra where the composition is unknown."""
    # input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
    # label_list = [
    #     "Ni2pCo2pFe2p Ni metal",
    #     "Ni2pCo2pFe2p NiO",
    #     "Ni2pCo2pFe2p Co metal",
    #     "Ni2pCo2pFe2p CoO",
    #     "Ni2pCo2pFe2p Co3O4",
    #     "Ni2pCo2pFe2p Fe metal",
    #     "Ni2pCo2pFe2p FeO",
    #     "Ni2pCo2pFe2p Fe3O4",
    #     "Ni2pCo2pFe2p Fe2O3",
    # ]
    # new_start, new_stop, new_step = 700.0, 890.0, 0.1
    # output_file = r"C:\Users\pielsticker\Simulations\20220721_NiCoFe.h5"

    # convert_multiple_measured_spectra(
    #     input_datafolder=input_datafolder,
    #     new_start=new_start,
    #     new_stop=new_stop,
    #     new_step=new_step,
    #     output_file=output_file,
    #     label_list=label_list,
    #     plot_all=True
    # )

    """Test one of the created files.."""
    # saved_file = ""
    # with h5py.File(saved_file, "r") as hf:
    # size = hf["X"].shape
    # X_h5 = hf["X"][:, :, :]
    # y_h5 = hf["y"][:, :]
    # names_h5 = hf["names"][:]
    # energies_h5 = hf["energies"][:]
    # labels_h5 = [str(label) for label in hf["labels"][:]]
