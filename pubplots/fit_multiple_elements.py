# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:13:54 2022

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import ParserWrapper

datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"

# %% lot of various different peak fitting methods
class MultipleElementWrapper(ParserWrapper):
    def __init__(self, datafolder, file_dict):
        super(MultipleElementWrapper, self).__init__(
            datafolder=datafolder, file_dict=file_dict
        )
        self.fontdict_small = {"size": 13}
        self.fontdict_legend = {"size": 14.5}

    def plot_all(
            self,
            with_fits=True,
            with_nn_col=False):
        super(MultipleElementWrapper, self).plot_all(with_fits, with_nn_col)

        texts = [
            ("Ni 2p", 0.065, 0.65),
            ("Co 2p", 0.4, 0.575),
            ("Fe 2p", 0.77, 0.4),
            ]

        for i, parser in enumerate(self.parsers):
            self.axs[0, i].set_title(parser.title, fontdict=self.fontdict)

            q1 = [f"{p[0]} %" for p in list(parser.quantification.values())]
            q2 = [f"{p[1]} %" for p in list(parser.quantification.values())]

            keys = list(parser.quantification.keys())
            col_labels = self._reformat_label_list(keys)

            table = self.axs[0, i].table(
                cellText=[q1,q2],
                cellLoc="center",
                colLabels=col_labels,
                rowLabels=["Peak fitting", "CNN"],
                bbox=[0.12, 0.79, 0.69, 0.15],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.fontdict_small["size"])

            self.axs[0, i].text(
                0.01,
                0.96,
                "Quantification:",
                horizontalalignment="left",
                size=self.fontdict_small["size"],
                verticalalignment="center",
                transform=self.axs[0, i].transAxes,
            )

            for t in texts:
                self.axs[0, i].text(
                    x=t[1],
                    y=t[2],
                    s=t[0],
                    horizontalalignment="left",
                    size=25,
                    verticalalignment="center",
                    transform=self.axs[0, i].transAxes
                )

            self.axs[0,i].set_ylim(
                top=np.max(parser.data["y"]))

        self.fig.tight_layout()

        return self.fig, self.axs


# %% Plot of spectrum with multiple elements
file_dict = {
    "true": {
        "filename": "nicofe_fit.txt",
        "title": "",
        "fit_start": 3,
        "fit_end": -1,
        "shift_x": 0.0,
        "noise": 32.567,
        "FWHM": 1.28,
        "scatterer": "N2",
        "distance": 0.18,
        "pressure": 0.3,
        "quantification": {
            "Ni metal": [1.8, 0.4],
            "NiO": [23.4, 30.5],
            "Co metal": [2.4, 0.0],
            "CoO": [3.5, 2.7],
            "Co3O4": [36.7, 28.9],
            "Fe metal": [0.1, 0.0],
            "FeO": [3.2, 3.0],
            "Fe3O4": [0.1, 2.2],
            "Fe2O3": [28.8, 32.3],
        },
    }
}

wrapper = MultipleElementWrapper(datafolder, file_dict)
wrapper.parse_data(bg=True, envelope=True)
fig, ax = wrapper.plot_all(
    with_fits=True,
    with_nn_col=False)
plt.show()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures"
fig_path = os.path.join(save_dir, "fit_multiple_elements.png")
fig.savefig(fig_path)
