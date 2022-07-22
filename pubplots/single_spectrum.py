# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:31:17 2022

@author: pielsticker
"""

import numpy as np
import os
import matplotlib.pyplot as plt


os.chdir(
    os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps")
)

from xpsdeeplearning.simulation.base_model.spectra import (
    MeasuredSpectrum)

datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"

#%% For one reference spectrum.

class Figure:
    """Class for plotting an XPS spectrum."""

    def __init__(self, x, y, title):
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
        fontdict = {"size": 25}
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(11.6, 3.8), dpi=300)
        self.fig.patch.set_facecolor("white")
        self.fig.patch.set_edgecolor("none")
        self.ax.plot(x, y, linewidth=3, color ="grey")
        self.ax.set_xlim(left=np.max(x), right=np.min(x))
      #  self.ax.set_xlabel("Binding energy (eV)", fontdict=fontdict)
        self.ax.set_axis_off()
#        self.ax.set_yticklabels([])
 #       self.ax.tick_params(axis="x", labelsize=fontdict["size"])
 #       self.ax.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
        self.ax.set_title(title, fontdict=fontdict)
        self.fig.tight_layout()


#%%

input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
filename = "Synthetic spectrum 17.vms"

filepath = os.path.join(input_datafolder, filename)
ref_spectrum = MeasuredSpectrum(filepath)
figure = Figure(x=ref_spectrum.x, y=ref_spectrum.lineshape, title="")

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures"
fig_filename = f"test_spectrum.tif"
fig_path = os.path.join(save_dir, fig_filename)
figure.fig.savefig(fig_path)