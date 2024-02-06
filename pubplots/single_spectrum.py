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
Plot a single Fe 2p spectrum.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from common import save_dir
from xpsdeeplearning.simulation.base_model.spectra import MeasuredSpectrum


class Figure:
    """Class for plotting an XPS spectrum."""

    def __init__(self, x, y, title, axis_off=True):
        """
        Plot the spectrum.

        X limits are set according to the min/max values of the
        binding energies.

        Parameters
        ----------
        x : ndarray
            Binding energy values.
        y : ndarray
            Intensity values.
        title : str
            Title of the plot.

        Returns
        -------
        None.

        """
        fontdict = {"size": 22}
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(11.6, 3.8), dpi=300)
        self.fig.patch.set_facecolor("white")
        self.fig.patch.set_edgecolor("none")
        self.ax.plot(x, y, linewidth=3, color="grey")
        self.ax.set_xlim(left=np.max(x), right=np.min(x))
        if axis_off:
            self.ax.set_axis_off()
        else:
            self.ax.set_xlabel("Binding energy (eV)", fontdict=fontdict)
            self.ax.set_yticklabels([])
            self.ax.tick_params(axis="x", labelsize=fontdict["size"])
            self.ax.set_ylabel("Intensity (arb. units)", fontdict=fontdict)

        self.ax.set_title(title, fontdict=fontdict)
        self.fig.tight_layout()


class FlippedFigure:
    """Class for plotting an XPS spectrum."""

    def __init__(self, x, y, title, axis_off=True):
        """
        Plot the spectrum.

        X limits are set according to the min/max values of the
        binding energies.

        Parameters
        ----------
        x : ndarray
            Binding energy values.
        y : ndarray
            Intensity values.
        title : str
            Title of the plot.

        Returns
        -------
        None.

        """
        fontdict = {"size": 22}
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(6, 11.6), dpi=300)
        self.fig.patch.set_facecolor("white")
        self.fig.patch.set_edgecolor("none")

        self.ax.scatter(y, x, c="black")  # cm.hot(y), edgecolor="none")

        self.ax.set_ylim(bottom=np.min(x), top=np.max(x))
        if axis_off:
            self.ax.set_axis_off()
        else:
            self.ax.set_ylabel("Binding energy (eV)", fontdict=fontdict)
            self.ax.set_xticklabels([])
            # for major ticks
            self.ax.set_xticks([])
            # for minor ticks
            self.ax.set_xticks([], minor=True)
            self.ax.tick_params(axis="y", labelsize=fontdict["size"])
            self.ax.set_xlabel("Intensity (arb. units)", fontdict=fontdict)

        self.ax.set_title(title, fontdict=fontdict)
        self.fig.tight_layout()


def main():
    input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
    filename = "fe2p_for_comparison_plot.vms"

    filepath = os.path.join(input_datafolder, filename)
    ref_spectrum = MeasuredSpectrum(filepath)
    figure = Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="", axis_off=True)
    figure2 = FlippedFigure(
        x=ref_spectrum.x, y=ref_spectrum.lineshape, title="", axis_off=False
    )

    file_names = {
        figure: "test_spectrum.tif",
        figure2: "test_spectrum_flipped.tif"
        }

    for fig, file_name in file_names.items():
        fig_path = os.path.join(save_dir, file_name)
        fig.fig.savefig(fig_path)

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
