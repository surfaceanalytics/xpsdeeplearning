# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:03:47 2020

@author: pielsticker


This script can be used to combine two datasets with data of the same
sahpe. A typical use case would be two data sets created by simulations
from different base reference spectra that shall be used as a combined
training/test data set.

"""
import os
import numpy as np
import h5py
from sklearn.utils import shuffle

#%%
def _load_data(filenames):
    """
    Load data from all HDF5 file and combine them into arrays.

    Parameters
    ----------
    filenames : list
        List of names of the files that are to be combined.

    Returns
    -------
    X : ndarray
        Features of the spectra.
    y : ndarray
        Labels of the spectra.
    shiftx : ndarray
        Energies of the horizontal shift of the spectra.
    noise : ndarray
        Signal-to-noise ratio of the artificially added noise.
    fwhm : ndarray
        Width for the artifical broadening.
    scatterer : ndarray
        Names of the scatterer. Encoded by integers.
    distance : ndarray
        Distances (in mm) the electrons travel in the scattering 
        medium.
    pressure : ndarray
        Pressure of the scattering medium in mbar.
    """
    input_datafolder = r"C:\Users\pielsticker\Simulations"
    for filename in filenames[:1]:
        input_filepath = os.path.join(input_datafolder, filename)
        with h5py.File(input_filepath, "r") as hf:
            X = hf["X"][:, :, :]
            y = hf["y"][:, :]
            shiftx = hf["shiftx"][:]
            noise = hf["noise"][:]
            fwhm = hf["FWHM"][:]
            scatterer = hf["scatterer"][:]
            distance = hf["distance"][:]
            pressure = hf["pressure"][:]
            energies = hf["energies"][:]
            labels = hf["labels"][:]
            print("File 0 loaded")

    for filename in filenames[1:]:
        input_filepath = os.path.join(input_datafolder, filename)
        with h5py.File(input_filepath, "r") as hf:
            X_new = hf["X"][:, :, :]
            y_new = hf["y"][:, :]
            shiftx_new = hf["shiftx"][:]
            noise_new = hf["noise"][:]
            fhwm_new = hf["FWHM"][:]
            scatterer_new = hf["scatterer"][:]
            distance_new = hf["distance"][:]
            pressure_new = hf["pressure"][:]

            X = np.concatenate((X, X_new), axis=0)
            y = np.concatenate((y, y_new), axis=0)
            shiftx = np.concatenate((shiftx, shiftx_new), axis=0)
            noise = np.concatenate((noise, noise_new), axis=0)
            fwhm = np.concatenate((fwhm, fhwm_new), axis=0)
            scatterer = np.concatenate(
                (scatterer, scatterer_new), axis=0
            )
            distance = np.concatenate((distance, distance_new), axis=0)
            pressure = np.concatenate((pressure, pressure_new), axis=0)
            print("File {0} loaded".format(filenames.index(filename)))

    return (
        X,
        y,
        shiftx,
        noise,
        fwhm,
        scatterer,
        distance,
        pressure,
        energies,
        labels,
    )


def combine_and_shuffle_measured(filenames, output_file):
    """
    Load the data from the given filenames.
    
    The data is shuffled and stored in a new file.

    Parameters
    ----------
    filenames : list
        List of names of the HDF5 files that are to be combined.
    output_file : str
        Filepath of the output file.

    Returns
    -------
    None.

    """
    (
        X,
        y,
        shiftx,
        noise,
        fwhm,
        scatterer,
        distance,
        pressure,
        energies,
        labels,
    ) = _load_data(filenames)

    # Shuffle all numpy arrays together.
    (
        X_shuff,
        y_shuff,
        shiftx_shuff,
        noise_shuff,
        fwhm_shuff,
        scatterer_shuff,
        distance_shuff,
        pressure_shuff,
    ) = shuffle(
        X, y, shiftx, noise, fwhm, scatterer, distance, pressure
    )

    with h5py.File(output_file, "w") as hf:
        hf.create_dataset(
            "X",
            data=X_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, X.shape[1], X.shape[2]),
        )
        print("X written")
        hf.create_dataset(
            "y",
            data=y_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, y.shape[1]),
        )
        print("y written")
        hf.create_dataset(
            "shiftx",
            data=shiftx_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, shiftx.shape[1]),
        )
        print("shift written")
        hf.create_dataset(
            "noise",
            data=noise_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, noise.shape[1]),
        )
        print("noise written")
        hf.create_dataset(
            "FWHM",
            data=fwhm_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, fwhm.shape[1]),
        )
        print("fwhm written")
        hf.create_dataset(
            "scatterer",
            data=scatterer_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, scatterer.shape[1]),
        )
        print("scatterer written")
        hf.create_dataset(
            "distance",
            data=distance_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, distance.shape[1]),
        )
        print("distance written")
        hf.create_dataset(
            "pressure",
            data=pressure_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, pressure.shape[1]),
        )
        print("pressure written")
        hf.create_dataset(
            "energies", data=energies, compression="gzip", chunks=True
        )
        print("energies written")
        hf.create_dataset(
            "labels", data=labels, compression="gzip", chunks=True
        )
        print("labels written")


#%%
if __name__ == "__main__":
    filenames = [
        "20210222_Fe_linear_combination_small_gas_phase.h5",
        "20210310_Fe_linear_combination_small_gas_phase_Mark.h5",
    ]
    output_file = r"C:\Users\pielsticker\Simulations\20210311_Fe_linear_combination_small_gas_phase_combined_data.h5"

combine_and_shuffle_measured(filenames, output_file)

# Test new file.
with h5py.File(output_file, "r") as hf:
    size = hf["X"].shape
    X_h5 = hf["X"][:100, :, :]
    y_h5 = hf["y"][:100, :]
    shiftx_h5 = hf["shiftx"][:100]
    noise_h5 = hf["noise"][:100]
    fwhm_h5 = hf["FWHM"][:100]
    energies_h5 = hf["energies"][:]
    labels_h5 = [str(label) for label in hf["labels"][:]]
