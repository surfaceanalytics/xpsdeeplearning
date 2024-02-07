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
Plot different peak fitting methods and their quantification on an artificial
Fe 2p XP spectrum.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import mean_absolute_error
import numpy as np
from PIL import Image

from common import ParserWrapper, DATAFOLDER, SAVE_DIR


class Wrapper(ParserWrapper):
    """Wrapper for loading and plotting."""

    def __init__(self, datafolder, file_dict):
        super().__init__(datafolder=datafolder, file_dict=file_dict)
        self.fontdict["size"] = 32
        self.fontdict_small["size"] = 18
        self.fontdict_legend["size"] = 16

        self.color_dict = {
            "CPS": "grey",
            "Fe metal": "black",
            "FeO": "green",
            "Fe3O4": "blue",
            "Fe2O3": "orange",
            "Fe sat": "deeppink",
            "Background": "tab:brown",
            "Envelope": "red",
        }

    def _add_nn_image(self, ax):
        """Add the image of a neural network to an axis."""
        ax.set_axis_off()
        ax.set_title("(e) Convolutional Neural network", fontdict=self.fontdict)

        infile = os.path.join(
            SAVE_DIR,
            "neural_net_peak_fitting.tif",
        )
        with Image.open(infile) as im:
            ax.imshow(im)

        return ax

    def plot_all(self):
        """Plot data and neural network image."""
        self.fig = plt.figure(figsize=(34, 16), dpi=300)

        ncols = 2
        gs = gridspec.GridSpec(
            nrows=2,
            ncols=ncols + 1,
            wspace=0.15,
            hspace=0.4,
        )

        # Set up 2 plots in first three row and 1 in last row.
        ax0_0 = self.fig.add_subplot(gs[0, 0])
        ax0_1 = self.fig.add_subplot(gs[1, 0])
        ax1_0 = self.fig.add_subplot(gs[0, 1])
        ax1_1 = self.fig.add_subplot(gs[1, 1])
        ax3 = self.fig.add_subplot(gs[:, 2])

        self.axs = np.array([[ax0_0, ax0_1, ax3], [ax1_0, ax1_1, None]])

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            row, col = int(i / ncols), i % ncols

            self.axs[row, col].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            self.axs[row, col].set_ylabel(
                "Intensity (arb. units)", fontdict=self.fontdict
            )
            self.axs[row, col].tick_params(axis="x", labelsize=self.fontdict["size"])
            self.axs[row, col].tick_params(
                axis="y", which="both", right=False, left=False
            )

            self.axs[row, col].set_yticklabels([])

            handle_dict = {}
            labels = []

            for j, header_name in enumerate(parser.header_names):
                for spectrum_name in parser.names:
                    if spectrum_name in header_name:
                        name = spectrum_name

                color = self.color_dict[name]
                start = parser.fit_start
                end = parser.fit_end
                if name == "CPS":
                    handle = self.axs[row, col].plot(x, y[:, j], c=color)
                else:
                    try:
                        handle = self.axs[row, col].plot(
                            x[start:end], y[start:end, j], c=color
                        )
                    except IndexError:
                        handle = self.axs[row, col].plot(
                            x[start:end], y[start:end], c=color
                        )

                if name not in handle_dict:
                    handle_dict[name] = handle
                    labels.append(name)

            handles = [
                x for handle_list in list(handle_dict.values()) for x in handle_list
            ]
            labels = self._reformat_label_list(labels)

            if i > 0:  # "spectrum.txt" not in self.parsers[i].filepath:
                self.axs[row, col].legend(
                    handles=handles,
                    labels=labels,
                    prop={"size": self.fontdict_legend["size"]},
                    loc="upper right",
                )
                self.axs[row, col].text(
                    0.5,
                    0.1,
                    f"MAE: {np.round(parser.mae, 3)}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    size=self.fontdict_small["size"],
                    transform=self.axs[row, col].transAxes,
                )

            self.axs[row, col].set_title(parser.title, fontdict=self.fontdict)
            self.axs[row, col].set_xlim(left=np.max(x), right=np.min(x))

            quantification = [f"{p} %" for p in list(parser.quantification.values())]

            if "spectrum.txt" not in self.parsers[i].filepath:
                quant_text = "Quantification:"
            else:
                quant_text = "Ground truth:"

            self.axs[row, col].text(
                0.03,
                0.225,
                quant_text,
                horizontalalignment="left",
                verticalalignment="center",
                size=self.fontdict_small["size"],
                transform=self.axs[row, col].transAxes,
            )

            keys = list(parser.quantification.keys())
            col_labels = self._reformat_label_list(keys)

            table = self.axs[row, col].table(
                cellText=[quantification],
                cellLoc="center",
                colLabels=col_labels,
                bbox=[0.03, 0.03, 0.45, 0.15],
                zorder=500,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.fontdict_small["size"])

        self.axs[0, -1] = self._add_nn_image(self.axs[0, -1])

        gs.tight_layout(self.fig)

        return self.fig


def main():
    """Plot of various different peak fitting methods."""

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
                "FeO": 17.5,
                "Fe3O4": 22.9,
                "Fe2O3": 29.5,
            },
        },
        "Peak fit model": {
            "filename": "fit_model.txt",
            "title": "(c) M2: Manual peak fit model",
            "fit_start": 50,
            "fit_end": -137,
            "quantification": {
                "Fe metal": 31.2,
                "FeO": 5.6,
                "Fe3O4": 8.4,
                "Fe2O3": 54.8,
            },
            "mae": 0.132,
        },
        "Biesinger": {
            "filename": "fit_biesinger.txt",
            "title": "(b) M1: Fit according to Grosvenor et al.",
            "fit_start": 645,
            "fit_end": -145,
            "quantification": {
                "Fe metal": 36.6,
                "FeO": 11.4,
                "Fe3O4": 34.8,
                "Fe2O3": 17.2,
            },
            "mae": 0.092,
        },
        "Tougaard lineshapes": {
            "filename": "lineshapes_tougaard.txt",
            "title": "(d) M3: Fit with reference lineshapes",
            "fit_start": 50,
            "fit_end": -137,
            "quantification": {
                "Fe metal": 35.0,
                "FeO": 15.9,
                "Fe3O4": 20.7,
                "Fe2O3": 28.4,
            },
            "mae": 0.02449,
        },
    }

    wrapper = Wrapper(DATAFOLDER, file_dict)
    wrapper.parse_data(bg=True, envelope=True)
    fig = wrapper.plot_all()
    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "peak_fitting_single" + ext)
        fig.savefig(fig_path, bbox_inches="tight")

    q_true = np.array(list(file_dict["true"]["quantification"].values())) / 100
    q_biesinger = (
        np.array(list(file_dict["Biesinger"]["quantification"].values())) / 100
    )
    q_fit_model = (
        np.array(list(file_dict["Peak fit model"]["quantification"].values())) / 100
    )
    q_lineshapes = (
        np.array(list(file_dict["Tougaard lineshapes"]["quantification"].values()))
        / 100
    )
    q_nn = np.array([29.9, 16.9, 23.4, 29.8]) / 100

    mae_biesinger = mean_absolute_error(q_true, q_biesinger)
    mae_fit_model = mean_absolute_error(q_true, q_fit_model)

    mae_lineshapes = mean_absolute_error(q_true, q_lineshapes)
    mae_nn = mean_absolute_error(q_true, q_nn)

    print(f"Biesinger: {mae_biesinger}")
    print(f"Fit model: {mae_fit_model}")
    print(f"Tougaard lineshapes: {mae_lineshapes}")
    print(f"CNN: {mae_nn}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
