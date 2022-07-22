# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:23:14 2022

@author: pielsticker
"""

import matplotlib.pyplot as plt
import os
from PIL import Image

from common import ParserWrapper

datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"

# %% Plot of various different peak fitting methods
class PeakFittingWrapper(ParserWrapper):
    def __init__(self, datafolder, file_dict):
        super(PeakFittingWrapper, self).__init__(
            datafolder=datafolder, file_dict=file_dict
        )
        self.fontdict_small = {"size": 14}

    def plot_all(
        self,
        with_fits=True,
        with_nn_col=True):
        super(PeakFittingWrapper, self).plot_all(with_fits, with_nn_col)

        for i, parser in enumerate(self.parsers):
            quantification = [
                f"{p} %" for p in list(parser.quantification.values())
                ]

            if "spectrum.txt" not in self.parsers[i].filepath:
                text = "Quantification:"
            else:
                text = "Ground truth:"

            self.axs[0,i].set_title(parser.title, fontdict=self.fontdict)

            quantification = [
                f"{p} %" for p in list(parser.quantification.values())
            ]

            table = self.axs[0,i].table(
                cellText=[quantification],
                cellLoc="center",
                colLabels=self.col_labels,
                bbox=[0.03, 0.03, 0.6, 0.15],
                )
            table.auto_set_font_size(False)
            table.set_fontsize(self.fontdict_small["size"])

            self.axs[0,i].text(
                0.03,
                0.215,
                text,
                horizontalalignment="left",
                size=self.fontdict_small["size"],
                verticalalignment="center",
                transform=self.axs[0,i].transAxes,
                )


            if with_nn_col:
                self.axs[0,-1].set_axis_off()
                self.axs[0,-1].set_title("(e) Neural Network", fontdict=self.fontdict)

                infile = os.path.join(
                    r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures",
                    "neural_net_multiple_elements.png")
                with Image.open(infile) as im:
                    self.axs[0,-1].imshow(im)

        self.fig.tight_layout()

        return self.fig, self.axs

file_dict = {
    "true": {
        "filename": "spectrum.txt",
        "title": "(a) Simulated spectrum",
        "fit_start": 0,
        "fit_end": -1,
        "shift_x": 0.0,
        "noise": 32.567,
        "FWHM": 1.28,
        "scatterer": "N2",
        "distance": 0.18,
        "pressure": 0.3,
        "quantification": {
            "Fe metal": 30.1,
            "FeO": 17.54,
            "Fe3O4": 22.9,
            "Fe2O3": 29.5,
        },
    },
    "Shirley lineshapes": {
        "filename": "shirley_fit.txt",
        "title": "(d) Peak model using a Shirley background",
        "fit_start": 50,
        "fit_end": -137,
        "quantification": {
            "Fe metal": 19.56,
            "FeO": 7.38,
            "Fe3O4": 7.26,
            "Fe2O3": 65.80,
        },
    },
    "Biesinger": {
        "filename": "biesinger.txt",
        "title": "(b) Fit according to Biesinger et al.",
        "fit_start": 645,
        "fit_end": -145,
        "quantification": {
            "Fe metal": 36.6,
            "FeO": 11.4,
            "Fe3O4": 34.8,
            "Fe2O3": 17.2,
        },
    },
    "Tougaard lineshapes": {
        "filename": "tougaard_lineshapes.txt",
        "title": "(c) Fit with reference lineshapes",
        "fit_start": 50,
        "fit_end": -137,
        "quantification": {
            "Fe metal": 38.0,
            "FeO": 11.9,
            "Fe3O4": 28.3,
            "Fe2O3": 21.7,
        },
    },
}

wrapper = PeakFittingWrapper(datafolder, file_dict)
wrapper.parse_data(bg=True, envelope=True)
fig, axs = wrapper.plot_all(
    with_fits=True,
    with_nn_col=True)
plt.show()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures"
fig_path = os.path.join(save_dir, "peak_fitting.png")
fig.savefig(fig_path)