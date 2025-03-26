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
#
"""
Figures for XY spectra.
"""
import matplotlib.pyplot as plt
import numpy as np


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
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=300)
        self.fig.patch.set_facecolor("0.9411")
        self.ax.plot(x, y)
        self.ax.set_xlim(left=np.max(x), right=np.min(x))
        self.ax.set_xlabel("Binding energy (eV)")
        self.ax.set_ylabel("Intensity (arb. units)")
        self.ax.set_title(title)
        self.fig.tight_layout()
