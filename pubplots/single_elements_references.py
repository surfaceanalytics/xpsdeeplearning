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
Plot reference spectra of different transition metals.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from common import TextParser, ParserWrapper, DATAFOLDER, SAVE_DIR


class FitTextParser(TextParser):
    """Parser for XPS data stored in TXT files."""

    def _build_data(self, background=True, envelope=True):
        """
        Build dictionary from the loaded data.

        Returns
        -------
        data_dict
            Data dictionary.

        """
        self.header_names = [
            hn.split(":")[1] for hn in self.header[6].split("\t") if "Cycle" in hn
        ]

        if background:
            self.header_names += ["Background"]
        if envelope:
            self.header_names += ["Envelope"]

        for i, header_name in enumerate(self.header_names):
            if header_name in ("Ni", "Co", "Fe"):
                self.header_names[i] += " metal"

        lines = np.array([[float(i) for i in d.split()] for d in self.data])

        x = lines[:, 0::2]
        y = lines[:, 1::2]

        y /= np.sum(y, axis=0)

        self.data = {"x": x, "y": y}

        return self.data


class Wrapper(ParserWrapper):
    """Wrapper for loading and plotting."""

    def __init__(self, datafolder, file_dict):
        super().__init__(datafolder=datafolder, file_dict=file_dict)
        self.fontdict["size"] = 30
        self.fontdict_legend["size"] = 24

        self.color_dict = {
            "CPS": "grey",
            "Cu metal": "black",
            "Cu2O": "crimson",
            "CuO": "green",
            "Co metal": "black",
            "CoO": "green",
            "Co3O4": "blue",
            "Fe metal": "black",
            "FeO": "green",
            "Fe3O4": "blue",
            "Fe2O3": "orange",
            "Ni metal": "black",
            "NiO": "green",
            "MnO": "green",
            "Mn2O3": "orange",
            "MnO2": "tab:purple",
            "Pd metal": "black",
            "PdO": "green",
            "Ti metal": "black",
            "TiO": "green",
            "Ti2O3": "orange",
            "TiO2": "tab:purple",
            "Background": "tab:brown",
            "Envelope": "red",
        }

    def parse_data(self, background=True, envelope=True):
        """Load data from file dict."""
        for element_dict in self.file_dict.values():
            filepath = os.path.join(self.datafolder, element_dict["filename"])
            parser = FitTextParser()
            parser.parse_file(filepath, background=background, envelope=envelope)
            for key, value in element_dict.items():
                setattr(parser, key, value)
            self.parsers.append(parser)

    def plot_all(self):
        """Plot reference spectra."""
        ncols = 2

        fig = plt.figure(figsize=(18, 24), dpi=300)
        gs = gridspec.GridSpec(4, ncols * 2, wspace=0.5, hspace=0.5)

        # Set up 2 plots in first three row and 1 in last row.
        ax0_0 = fig.add_subplot(gs[0, :2])
        ax0_1 = fig.add_subplot(gs[0, 2:])
        ax1_0 = fig.add_subplot(gs[1, :2])
        ax1_1 = fig.add_subplot(gs[1, 2:])
        ax2_0 = fig.add_subplot(gs[2, :2])
        ax2_1 = fig.add_subplot(gs[2, 2:])
        ax3 = fig.add_subplot(gs[3, 1:3])

        axs = np.array([[ax0_0, ax0_1], [ax1_0, ax1_1], [ax2_0, ax2_1], [ax3, None]])

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            row, col = int(i / ncols), i % ncols

            axs[row, col].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            axs[row, col].set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
            axs[row, col].tick_params(axis="x", labelsize=self.fontdict["size"])
            axs[row, col].tick_params(axis="y", which="both", right=False, left=False)

            axs[row, col].set_yticklabels([])

            handle_dict = {}
            labels = []

            for j, header_name in enumerate(parser.header_names):
                for spectrum_name in parser.names:
                    if spectrum_name in header_name:
                        name = spectrum_name

                color = self.color_dict[name]

                start = parser.fit_start
                end = parser.fit_end

                handle = axs[row, col].plot(
                    x[start:end, j], y[start:end, j], c=color, linewidth=2
                )

                if name not in handle_dict:
                    handle_dict[name] = handle
                    labels.append(name)

                handles = [
                    x for handle_list in list(handle_dict.values()) for x in handle_list
                ]
                labels = self._reformat_label_list(labels)

                axs[row, col].set_xlim(left=x[start, j], right=x[end, j])

            axs[row, col].legend(
                handles=handles,
                labels=labels,
                ncol=1,
                prop={"size": self.fontdict_legend["size"]},
                loc="upper left",
            )

            axs[row, col].set_title(parser.title, fontdict=self.fontdict)

        gs.tight_layout(fig)

        return fig


def main():
    """Plot referene spectra of different transition metals."""

    file_dict = {
        "Co": {
            "filename": "references_co.txt",
            "title": "(a) Co 2p",
            "fit_start": 1,
            "fit_end": -1,
        },
        "Cu": {
            "filename": "references_cu.txt",
            "title": "(b) Cu 2p",
            "fit_start": 0,
            "fit_end": -1,
        },
        "Fe": {
            "filename": "references_fe.txt",
            "title": "(c) Fe 2p",
            "fit_start": 0,
            "fit_end": -20,
        },
        "Mn": {
            "filename": "references_mn.txt",
            "title": "(d) Mn 2p",
            "fit_start": 0,
            "fit_end": -1,
        },
        "Ni": {
            "filename": "references_ni.txt",
            "title": "(e) Ni 2p",
            "fit_start": 1,
            "fit_end": -1,
        },
        "Pd": {
            "filename": "references_pd.txt",
            "title": "(f) Pd 3d",
            "fit_start": 0,
            "fit_end": -1,
        },
        "Ti": {
            "filename": "references_ti.txt",
            "title": "(g) Ti 2p",
            "fit_start": 20,
            "fit_end": -1,
        },
    }

    wrapper = Wrapper(DATAFOLDER, file_dict)
    wrapper.parse_data(background=False, envelope=False)
    fig = wrapper.plot_all()

    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "references_single" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
