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
This script can be used to find those spectra in a HDF5 file that contain
peaks (and are thus suitable for deep learning XPS spectra) and create a
new file only containing these spectra.
"""
import h5py
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

from xpsdeeplearning.simulation.base_model.figures import Figure


class Peakfinder:
    """Class for finding peaks in a spectrum."""

    def __init__(self, X, y):
        """
        Initialize data and labels.

        Parameters
        ----------
        X : array
            Array with intensity data, should be of size
            (no_of_spectra, no_of_datapoints, 1).
        y : array
            Array with labels , should be of size
            (no_of_spectra, 1).

        Returns
        -------
        None.

        """
        self.X = X
        self.y = y

    def get_peak_positions(self, prominence):
        """
        Get all peaks in all spectra.

        Parameters
        ----------
        prominence: number or ndarray or sequence, optional
            Required prominence of peaks.
            Either a number, None, an array matching x or a 2-element
            sequence of the former. The first element is always
            interpreted as the minimal and the second, if supplied, as
            the maximal required prominence.

        Returns
        -------
        list
            List of arrays containing all peak positions.

        """
        self.all_peaks = []
        for x in self.X:
            x = np.squeeze(x)
            peaks, _ = find_peaks(x, prominence=prominence)
            self.all_peaks.append(peaks)

        return self.all_peaks

    def get_indices(self):
        """
        Get indices of spectra with and without peaks.

        Returns
        -------
        None.

        """
        self.peakfull_indices = np.where([len(p) > 0 for p in self.all_peaks])
        self.peakless_indices = np.where([len(p) == 0 for p in self.all_peaks])

    def distinguish_arrays(self, array_dict):
        """
        Distinguish between array with and without peaks.

        Parameters
        ----------
        array_dict : dict, optional
            Dictionaries with additional descriptive arrays
            that should also be split.

        Returns
        -------
        arrays_with_peaks : dict
            Dictionary with the spectra that have peaks.
        arrays_without_peak : dict
            Dictionary with the spectra that do not have peaks.

        """
        X_with_peaks = self.X[self.peakfull_indices]
        y_with_peaks = self.y[self.peakfull_indices]
        X_without_peaks = self.X[self.peakless_indices]
        y_without_peaks = self.y[self.peakless_indices]

        self.arrays_with_peaks = {"X": X_with_peaks, "y": y_with_peaks}
        self.arrays_without_peaks = {
            "X": X_without_peaks,
            "y": y_without_peaks,
        }

        for key, array in array_dict.items():
            self.arrays_with_peaks[key] = array[self.peakfull_indices]
            self.arrays_without_peaks[key] = array[self.peakless_indices]

        return self.arrays_with_peaks, self.arrays_without_peaks

    def plot_random_spectra(self, energies, no_of_spectra=5, with_peaks=True):
        """
        Plot random spectra with indicated peaks.

        Parameters
        ----------
        energies : list or 1D array
            Values for the x axis.
        no_of_spectra : int, optional
            No. of random spectra to plot. The default is 5.
        with_peaks : bool, optional
            If with_peaks, only spectra with peaks are shown.
            Else, only spectra without peaks are shown.
            The default is True.

        Returns
        -------
        None.

        """
        x = energies

        for _ in range(no_of_spectra):
            if with_peaks:
                p = self.peakfull_indices[0][
                    np.random.randint(0, self.peakfull_indices[0].shape[0])
                ]
                peaks = self.all_peaks[p]
                spectrum_text = "With peaks"
            else:
                p = self.peakless_indices[0][
                    np.random.randint(0, self.peakless_indices[0].shape[0])
                ]
                spectrum_text = "No peaks"
                peaks = []

            title = "Simulated spectrum no. " + str(p)
            y = self.X[p]
            fig = Figure(x, y, title)
            fig.ax.scatter(x=energies[peaks], y=y[peaks], s=25, c="red")
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

    def write_new_file(self, original_filepath):
        """
        Write only the data with peaks to a new file.

        Parameters
        ----------
        original_filepath : str
            Filepath of the original HDF5 file.

        Returns
        -------
        None.

        """
        with h5py.File(original_filepath, "r") as hf:
            energies_h5 = hf["energies"][:]
            labels_h5 = [str(label) for label in hf["labels"]]

        hdf5_data = self.arrays_with_peaks
        self.arrays_with_peaks["energies"] = energies_h5
        self.arrays_with_peaks["labels"] = labels_h5

        new_filepath = original_filepath.split(".h5", 1)[0] + "_peaks_only.h5"

        with h5py.File(new_filepath, "w") as hf:
            for key, value in hdf5_data.items():
                try:
                    hf.create_dataset(key, data=value, compression="gzip", chunks=True)
                except TypeError:
                    value = np.array(value, dtype=object)
                    string_dt = h5py.special_dtype(vlen=str)
                    hf.create_dataset(
                        key,
                        data=value,
                        dtype=string_dt,
                        compression="gzip",
                        chunks=True,
                    )


if __name__ == "__main__":
    # Data loading
    hdf5_filepath = r"C:\Users\pielsticker\Simulations\20210915_CoFe_combined_with_auger_peaks_100eV_window\20210915_CoFe_combined_with_auger_peaks_100eV_window.h5"
    with h5py.File(hdf5_filepath, "r") as hf:
        sizes_h5 = [(key, hf[key].shape) for key in list(hf.keys())]
        X_h5 = hf["X"][:200000, :, :]
        y_h5 = hf["y"][:200000, :]
        shiftx_h5 = hf["shiftx"][:200000, :]
        noise_h5 = hf["noise"][:200000, :]
        FWHM_h5 = hf["FWHM"][200000:, :]
        scatterer_h5 = hf["scatterer"][:200000, :]
        pressure_h5 = hf["pressure"][200000:, :]
        distance_h5 = hf["distance"][:200000, :]
        energies_h5 = hf["energies"][:]
        labels_h5 = [str(label) for label in hf["labels"]]

    peakfinder = Peakfinder(X_h5, y_h5)
    all_peaks = peakfinder.get_peak_positions(prominence=0.000025)
    peakfinder.get_indices()
    print(f"Spectra with peaks: {len(peakfinder.peakfull_indices[0])}")
    print(f"Spectra without peaks: {len(peakfinder.peakless_indices[0])}")
    peakfinder.plot_random_spectra(energies_h5, no_of_spectra=10, with_peaks=True)
    peakfinder.plot_random_spectra(energies_h5, no_of_spectra=10, with_peaks=False)

    array_dict = {
        "shiftx": shiftx_h5,
        "noise": noise_h5,
        "FWHM": FWHM_h5,
        "scatterer": scatterer_h5,
        "pressure": pressure_h5,
        "distance": shiftx_h5,
    }

    arrays_with_peaks, arrays_without_peak = peakfinder.distinguish_arrays(array_dict)
    peakfinder.write_new_file(original_filepath=hdf5_filepath)

    new_filepath = r"C:\Users\pielsticker\Simulations\20210915_CoFe_individual_with_auger_peaks_35eV_window\20210915_CoFe_individual_with_auger_peaks_35eV_window_peaks_only.h5"
    with h5py.File(new_filepath, "r") as hf:
        sizes_peaks = [(key, hf[key].shape) for key in list(hf.keys())]
        X_peaks = hf["X"][:20, :, :]
        y_peaks = hf["y"][:20, :]
        shiftx_peaks = hf["shiftx"][:20, :]
        noise_peaks = hf["noise"][:20, :]
        FWHM_peaks = hf["FWHM"][:20, :]
        scatterer_peaks = hf["scatterer"][:20, :]
        pressure_peaks = hf["pressure"][:20, :]
        distance_peaks = hf["distance"][:20, :]
        energies_peaks = hf["energies"][:20]
        labels_peaks = [str(label) for label in hf["labels"]]
