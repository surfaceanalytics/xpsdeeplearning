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
This script can be used to combine two datasets with data of the same
shape. A typical use case would be two data sets created by simulations
from different base reference spectra that shall be used as a combined
training/test data set.
"""
import os
import numpy as np
import h5py
from sklearn.utils import shuffle


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
        with h5py.File(input_filepath, "r") as h5_file:
            X = h5_file["X"][:, :, :]
            y = h5_file["y"][:, :]
            shiftx = h5_file["shiftx"][:]
            noise = h5_file["noise"][:]
            fwhm = h5_file["FWHM"][:]
            scatterer = h5_file["scatterer"][:]
            distance = h5_file["distance"][:]
            pressure = h5_file["pressure"][:]
            energies = h5_file["energies"][:]
            labels = h5_file["labels"][:]
            print("File 0 loaded")

    for filename in filenames[1:]:
        input_filepath = os.path.join(input_datafolder, filename)
        with h5py.File(input_filepath, "r") as h5_file:
            X_new = h5_file["X"][:, :, :]
            y_new = h5_file["y"][:, :]
            shiftx_new = h5_file["shiftx"][:]
            noise_new = h5_file["noise"][:]
            fhwm_new = h5_file["FWHM"][:]
            scatterer_new = h5_file["scatterer"][:]
            distance_new = h5_file["distance"][:]
            pressure_new = h5_file["pressure"][:]

            X = np.concatenate((X, X_new), axis=0)
            y = np.concatenate((y, y_new), axis=0)
            shiftx = np.concatenate((shiftx, shiftx_new), axis=0)
            noise = np.concatenate((noise, noise_new), axis=0)
            fwhm = np.concatenate((fwhm, fhwm_new), axis=0)
            scatterer = np.concatenate((scatterer, scatterer_new), axis=0)
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
    ) = shuffle(X, y, shiftx, noise, fwhm, scatterer, distance, pressure)

    with h5py.File(output_file, "w") as h5_file:
        h5_file.create_dataset(
            "X",
            data=X_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, X.shape[1], X.shape[2]),
        )
        print("X written")
        h5_file.create_dataset(
            "y",
            data=y_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, y.shape[1]),
        )
        print("y written")
        h5_file.create_dataset(
            "shiftx",
            data=shiftx_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, shiftx.shape[1]),
        )
        print("shift written")
        h5_file.create_dataset(
            "noise",
            data=noise_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, noise.shape[1]),
        )
        print("noise written")
        h5_file.create_dataset(
            "FWHM",
            data=fwhm_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, fwhm.shape[1]),
        )
        print("fwhm written")
        h5_file.create_dataset(
            "scatterer",
            data=scatterer_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, scatterer.shape[1]),
        )
        print("scatterer written")
        h5_file.create_dataset(
            "distance",
            data=distance_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, distance.shape[1]),
        )
        print("distance written")
        h5_file.create_dataset(
            "pressure",
            data=pressure_shuff,
            compression="gzip",
            chunks=True,
            maxshape=(None, pressure.shape[1]),
        )
        print("pressure written")
        h5_file.create_dataset(
            "energies", data=energies, compression="gzip", chunks=True
        )
        print("energies written")
        h5_file.create_dataset("labels", data=labels, compression="gzip", chunks=True)
        print("labels written")


if __name__ == "__main__":
    filenames = [
        "20210222_Fe_linear_combination_small_gas_phase.h5",
        "20210310_Fe_linear_combination_small_gas_phase_Mark.h5",
    ]
    output_file = r"C:\Users\pielsticker\Simulations\20210311_Fe_linear_combination_small_gas_phase_combined_data.h5"

    combine_and_shuffle_measured(filenames, output_file)

    # Test new file.
    with h5py.File(output_file, "r") as h5_file:
        size = h5_file["X"].shape
        X_h5 = h5_file["X"][:100, :, :]
        y_h5 = h5_file["y"][:100, :]
        shiftx_h5 = h5_file["shiftx"][:100]
        noise_h5 = h5_file["noise"][:100]
        fwhm_h5 = h5_file["FWHM"][:100]
        energies_h5 = h5_file["energies"][:]
        labels_h5 = [str(label) for label in h5_file["labels"][:]]
