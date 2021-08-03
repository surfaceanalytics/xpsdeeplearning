# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:43:16 2020.

@author: pielsticker
"""

import matplotlib.pyplot as plt
import numpy as np

#%%
class Figure:
    """Class for plotting an XPS spectrum."""

    def __init__(self, x, y, title):
        """
        Plot the spectrum.
        
        X limits are set according to the min/max values of the
        binding energies.

        Parameters
        ----------
        x : ndarray or list of ndarray
            Binding energy values.
        y : ndarray or list of ndarray
            Intensity values.
        title : str
            Title of the plot.

        Returns
        -------
        None.

        """
        if type(x) is not list:
            x = [x]
        if type(y) is not list:
            y = [y]
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor("0.9411")
        for (xi, yi) in zip(self.x, self.y):
            self.ax.plot(xi, yi)
        self.ax.set_xlim(left=np.max(x), right=np.min(x))
        self.ax.set_xlabel("Binding energy (eV)")
        self.ax.set_ylabel("Intensity (arb. units)")
        self.ax.set_title(title)
        self.fig.tight_layout()
