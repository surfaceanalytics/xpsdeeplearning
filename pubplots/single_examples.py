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
Plot example quantification on single-element XP spectra.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import TextParser, ParserWrapper, DATAFOLDER, SAVE_DIR


class Wrapper(ParserWrapper):
    """Wrapper for loading and plotting."""

    def __init__(self, datafolder, file_dict):
        super().__init__(datafolder=datafolder, file_dict=file_dict)
        self.fontdict_small = {"size": 14}
        self.fontdict_mae = {"size": 14}

    def parse_data(self, background=True, envelope=True):
        """Load data from file dict."""
        for result_dict in self.file_dict.values():
            for spectrum_dict in result_dict.values():
                filepath = os.path.join(self.datafolder, spectrum_dict["filename"])
                parser = TextParser()
                parser.parse_file(filepath, background=background, envelope=envelope)
                for key, value in spectrum_dict.items():
                    setattr(parser, key, value)
                self.parsers.append(parser)

    def plot_all(self):
        """Plot examples with predictions."""
        nrows, ncols = 2, 3

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(len(self.parsers) / 2 * 8, 14),
            squeeze=False,
            gridspec_kw={"hspace": 0.4, "wspace": 0.2},
            dpi=300,
        )

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            if i < len(self.parsers) / 2:
                color = "green"
            else:
                color = "red"

            if i < len(self.parsers) - 3:
                row = 0
            else:
                row = 1

            col = i % ncols

            axs[row, col].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            axs[row, col].set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
            axs[row, col].tick_params(axis="x", labelsize=self.fontdict["size"])
            axs[row, col].tick_params(axis="y", which="both", right=False, left=False)

            axs[row, col].set_yticklabels([])

            for j, header_name in enumerate(parser.header_names):
                if header_name == "CPS":
                    _ = axs[row, col].plot(x, y[:, j], c=color)
                else:
                    start = parser.fit_start
                    end = parser.fit_end
                    try:
                        _ = axs[row, col].plot(x[start:end], y[start:end, j], c=color)
                    except IndexError:
                        _ = axs[row, col].plot(x[start:end], y[start:end], c=color)

            axs[row, col].set_title(parser.title, fontdict=self.fontdict)
            axs[row, col].set_xlim(left=np.max(x), right=np.min(x))
            if row == 1 and col == 0:
                axs[row, col].set_ylim(bottom=np.min(y) * 0.999)

            axs[row, col].set_title(parser.title, fontdict=self.fontdict)

            quant_1 = [p[0] for p in list(parser.quantification.values())]
            quant_2 = [p[1] for p in list(parser.quantification.values())]

            q1_percentage = [f"{p} %" for p in quant_1]
            q2_percentage = [f"{p} %" for p in quant_2]

            keys = list(parser.quantification.keys())
            col_labels = self._reformat_label_list(keys)

            table = axs[row, col].table(
                cellText=[q1_percentage, q2_percentage],
                cellLoc="center",
                colLabels=col_labels,
                rowLabels=["Truth", "CNN"],
                bbox=[0.125, 0.025, 0.16 * len(quant_1), 0.18],
                zorder=500,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.fontdict_small["size"])

            axs[row, col].text(
                0.01,
                0.245,
                "Quantification:",
                horizontalalignment="left",
                size=self.fontdict_small["size"],
                verticalalignment="center",
                transform=axs[row, col].transAxes,
            )

            mae = f"MAE:\n{np.round(parser.mae,3)}"
            axs[row, col].text(
                x=0.85,
                y=0.85,
                s=mae,
                size=self.fontdict_mae["size"],
                verticalalignment="center",
                transform=axs[row, col].transAxes,
            )

            scatterer_dict = {
                "He": "He",
                "H2": "H$_{2}$",
                "N2": "N$_{2}$",
                "O2": "O$_{2}$",
            }
            try:
                scat_text = f"Scattered in {scatterer_dict[parser.scatterer]} \n"
                scat_text += f"p = {parser.pressure} mbar \n"
                scat_text += f"d = {parser.distance} mm"
                axs[row, col].text(
                    0.6,
                    0.5,
                    scat_text,
                    horizontalalignment="left",
                    size=self.fontdict_mae["size"],
                    verticalalignment="center",
                    transform=axs[row, col].transAxes,
                )
            except AttributeError:
                pass

        fig.tight_layout()

        return fig


def main():
    """Plot example quantification on single-element XP spectra."""
    file_dict = {
        "sucesses": {
            "Fe": {
                "filename": "fe2p_good_fit.txt",
                "title": "(a) Fe 2p",
                "fit_start": 0,
                "fit_end": -1,
                # "shift_x": -1.45,
                # "noise": 12.3,
                # "FWHM": 0.7,
                # "scatterer": "O2",
                # "distance": 0.3,
                # "pressure": 0.4,
                "mae": 0.004027272850268222,
                "quantification": {
                    "Fe metal": [25.2, 24.9],
                    "FeO": [27.0, 26.5],
                    "Fe3O4": [30.7, 30.8],
                    "Fe2O3": [17.1, 17.8],
                },
            },
            "Mn": {
                "filename": "mn2p_good_fit.txt",
                "title": "(b) Mn 2p",
                "fit_start": 0,
                "fit_end": -1,
                # "shift_x": 0.0,
                # "noise": 21.0,
                # "FWHM": 1.06,
                # "scatterer": "H2",
                # "distance": 0.4,
                # "pressure": 0.4,
                "mae": 0.004019161666664384,
                "quantification": {
                    "MnO": [31.5, 32.1],
                    "Mn2O3": [29.0, 28.3],
                    "MnO2": [39.5, 39.6],
                },
            },
            "Ti": {
                "filename": "ti2p_good_fit.txt",
                "title": "(c) Ti 2p",
                "fit_start": 0,
                "fit_end": -1,
                # "shift_x": 0.90,
                # "noise": 30.0,
                # "FWHM": 1.14,
                # "scatterer": "N2",
                # "distance": 0.3,
                # "pressure": 0.9,
                "mae": 0.00600266148683188,
                "quantification": {
                    "Ti metal": [24.5, 25.0],
                    "TiO": [19.8, 20.5],
                    "Ti2O3": [27.2, 26.7],
                    "TiO2": [28.4, 27.8],
                },
            },
            # =============================================================================
            #         "CoFe": {
            #             "filename": "cofe_good_fit.txt",
            #             "title": "(d) Mixed Co 2p and Fe 2p",
            #             "fit_start": 0,
            #             "fit_end": -1,
            #             # "shift_x": 0.95,
            #             # "noise": 18.0,
            #             # "FWHM": "not changed",
            #             # "scatterer": None,
            #             # "distance": 0.0,
            #             # "pressure": 0.0,
            #             "mae": 0.0,
            #             "quantification": {
            #                 "Co metal": [16.0, 15.6],
            #                 "CoO": [12.7, 14.0],
            #                 "Co3O4": [10.4, 11.6],
            #                 "Fe metal": [16.9, 14.2],
            #                 "FeO": [11.4, 11.7],
            #                 "Fe3O4": [16.9, 15.3],
            #                 "Fe2O3": [15.7, 17.4],
            #             },
            #         },
            # =============================================================================
        },
        "fails": {
            "fe2p_scattering": {
                "filename": "fe2p_strong_scattering.txt",
                "title": "(d) Fe 2p, strong scattering",
                "fit_start": 0,
                "fit_end": -1,
                # "shift_x": -2.85,
                # "noise": 13.8,
                # "FWHM": 0.89,
                # "scatterer": "N2",
                "distance": 2.0,
                "pressure": 4.3,
                "mae": 0.4123186282813549,
                "quantification": {
                    "Fe metal": [0.0, 72.1],
                    "FeO": [0.0, 10.3],
                    "Fe3O4": [51.3, 2.6],
                    "Fe2O3": [48.7, 15.0],
                },
            },
            "fe2p_noise": {
                "filename": "fe2p_noise.txt",
                "title": "(e) Fe 2p, low S/N",
                "fit_start": 0,
                "fit_end": -1,
                # "shift_x": -2.75,
                # "noise": 3.5,
                # "FWHM": 1.63,
                # "scatterer": "H2",
                # "distance": 0.9,
                # "pressure": 0.1,
                "mae": 0.29266935600171756,
                "quantification": {
                    "Fe metal": [0.0, 14.7],
                    "FeO": [61.6, 24.4],
                    "Fe3O4": [14.3, 58.1],
                    "Fe2O3": [24.1, 2.8],
                },
            },
            "mn_oxides": {
                "filename": "mn2p_similar_lineshapes.txt",
                "title": "(f) Mn 2p, similar peak shapes",
                "fit_start": 0,
                "fit_end": -1,
                # "shift_x": 0.90,
                # "noise": 3.2,
                # "FWHM": 0.55,
                # "scatterer": "O2",
                # "distance": 0.1,
                # "pressure": 0.5,
                "mae": 0.4059938986397533,
                "quantification": {
                    "MnO": [0.0, 0.3],
                    "Mn2O3": [23.3, 83.9],
                    "MnO2": [76.7, 15.8],
                },
            },
            # =============================================================================
            #         "cofe_mixed": {
            #             "filename": "cofe_bad_fit.txt",
            #             "title": "(g) Mixed Co 2p and Fe 2p",
            #             "fit_start": 0,
            #             "fit_end": -1,
            #             # "shift_x": 2.70,
            #             # "noise": 6.4,
            #             # "FWHM": "not changed",
            #             # "scatterer": None,
            #             # "distance": 0.0,
            #             # "pressure": 0.0,
            #             "mae": 0.08289022088380008,
            #             "quantification": {
            #                 "Co metal": [11.9, 24.7],
            #                 "CoO": [17.8, 18.1],
            #                 "Co3O4": [0.0, 0.0],
            #                 "Fe metal": [0.0, 0.6],
            #                 "FeO": [15.3, 17.1],
            #                 "Fe3O4": [0.0, 13.6],
            #                 "Fe2O3": [55.0, 26.0],
            #             },
            #         },
            # =============================================================================
        },
    }

    wrapper = Wrapper(DATAFOLDER, file_dict)
    wrapper.parse_data(background=False, envelope=False)
    fig = wrapper.plot_all()
    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "examples_single" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
