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
Plot of reference spectra with more than one element.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import TextParser, ParserWrapper, DATAFOLDER, SAVE_DIR


class FitTextParser(TextParser):
    """Parser for XPS data stored in TXT files."""

    def _build_data(self):
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

        for i, header_name in enumerate(self.header_names):
            if header_name in ("Ni", "Co", "Fe"):
                self.header_names[i] += " metal"

        self.names = [
            "CPS",
            "Co metal",
            "CoO",
            "Co3O4",
            "Fe metal",
            "FeO",
            "Fe3O4",
            "Fe2O3",
            "Fe sat",
            "Ni metal",
            "NiO",
        ]
        lines = np.array([[float(i) for i in d.split()] for d in self.data])

        x = lines[:, 0]
        y = lines[:, 1::2]

        y /= np.sum(y, axis=0)

        self.data = {"x": x, "y": y}

        return self.data


class Wrapper(ParserWrapper):
    """Wrapper for loading and plotting."""

    def __init__(self, datafolder, file_dict):
        super().__init__(datafolder=datafolder, file_dict=file_dict)
        self.fontdict_legend = {"size": 17}

        self.color_dict = {
            "CPS": "grey",
            "Co metal": "black",
            "CoO": "green",
            "Co3O4": "blue",
            "Fe metal": "black",
            "FeO": "green",
            "Fe3O4": "blue",
            "Fe2O3": "orange",
            "Ni metal": "black",
            "NiO": "green",
            "Background": "tab:brown",
            "Envelope": "red",
        }

        self.parser = FitTextParser()

    def parse_data(self):
        """Load data from file dict."""
        filepath = os.path.join(self.datafolder, self.file_dict["filename"])
        self.parser.parse_file(filepath)
        for key, value in self.file_dict.items():
            setattr(self.parser, key, value)

    def plot_all(self):
        """Plot references."""
        fig, axs = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(10, 12),
            squeeze=False,
            dpi=300,
        )

        x, y = self.parser.data["x"], self.parser.data["y"]

        handle_dict = {}
        labels = []

        for j, header_name in enumerate(self.parser.header_names):
            if "Ni" in header_name:
                row = 0

            elif "Co" in header_name:
                row = 1

            elif "Fe" in header_name:
                row = 2

            axs[row, 0].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            axs[row, 0].set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
            axs[row, 0].tick_params(axis="x", labelsize=self.fontdict["size"])
            axs[row, 0].tick_params(axis="y", which="both", right=False, left=False)
            axs[row, 0].set_yticklabels([])

            for spectrum_name in self.parser.names:
                if spectrum_name in header_name:
                    name = spectrum_name

            color = self.color_dict[name]

            start = self.parser.fit_start
            end = self.parser.fit_end

            handle = axs[row, 0].plot(x[start:end], y[start:end, j], c=color)

            axs[row, 0].set_xlim(left=np.max(x), right=np.min(x))

            if name not in handle_dict:
                handle_dict[name] = handle
                labels.append(name)

        handles = [x for handle_list in list(handle_dict.values()) for x in handle_list]
        labels = self._reformat_label_list(labels)

        # Only for this plot: Sort by oxidation state
        labels_sort_dict = {
            0: ["Ni metal", "NiO"],
            1: ["Co metal", "CoO", "Co$_{3}$O$_{4}$"],
            2: ["Fe metal", "FeO", "Fe$_{3}$O$_{4}$", "Fe$_{2}$O$_{3}$"],
        }

        for key, label_list in labels_sort_dict.items():
            handles_sort = []
            labels_sort = []
            for label in label_list:
                index = labels.index(label)
                handles_sort.append(handles[index])
                labels_sort.append(label)

            axs[key, 0].legend(
                handles=handles_sort,
                labels=labels_sort,
                ncol=1,
                prop={"size": self.fontdict_legend["size"]},
                loc="best",
            )

        fig.tight_layout()

        return fig


def main():
    """Plot of reference data with multiple elements."""

    file_dict = {
        "filename": "nicofe_references.txt",
        "title": "",
        "fit_start": 0,
        "fit_end": -1,
    }

    wrapper = Wrapper(DATAFOLDER, file_dict)
    wrapper.parse_data()
    fig = wrapper.plot_all()

    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "references_multiple" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
