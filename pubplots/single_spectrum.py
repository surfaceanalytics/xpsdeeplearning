# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:31:17 2022

@author: pielsticker
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

os.chdir(
    os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
)

from xpsdeeplearning.simulation.base_model.spectra import MeasuredSpectrum

datafolder = (
    r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
)

#%% For one reference spectrum.
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

        self.ax.scatter(y, x, c=cm.hot(y), edgecolor="none")

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


#%%
input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
filename = "fe2p_for_comparison_plot.vms"

filepath = os.path.join(input_datafolder, filename)
ref_spectrum = MeasuredSpectrum(filepath)
figure = Figure(
    x=ref_spectrum.x, y=ref_spectrum.lineshape, title="", axis_off=True
)
figure2 = FlippedFigure(
    x=ref_spectrum.x, y=ref_spectrum.lineshape, title="", axis_off=False
)


save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\figures"
fig_filename = "test_spectrum.tif"
fig_path = os.path.join(save_dir, fig_filename)
figure.fig.savefig(fig_path)
