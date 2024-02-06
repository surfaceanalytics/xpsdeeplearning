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

from common import TextParser, ParserWrapper, save_dir


class FitTextParser(TextParser):
    """Parser for XPS data stored in TXT files."""

    def _build_data(self, bg=True, envelope=True):
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

        for i, hn in enumerate(self.header_names):
            if hn == "Ni" or hn == "Co" or hn == "Fe":
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
    def __init__(self, datafolder, file_dict):
        super(Wrapper, self).__init__(datafolder=datafolder, file_dict=file_dict)
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

    def parse_data(self):
        filepath = os.path.join(self.datafolder, self.file_dict["filename"])
        self.parser = FitTextParser()
        self.parser.parse_file(filepath)
        self.parser.title = self.file_dict["title"]
        self.parser.fit_start = self.file_dict["fit_start"]
        self.parser.fit_end = self.file_dict["fit_end"]

    def plot_all(self):
        self.fig, self.axs = plt.subplots(
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

            self.axs[row, 0].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            self.axs[row, 0].set_ylabel(
                "Intensity (arb. units)", fontdict=self.fontdict
            )
            self.axs[row, 0].tick_params(axis="x", labelsize=self.fontdict["size"])
            self.axs[row, 0].tick_params(
                axis="y", which="both", right=False, left=False
            )
            self.axs[row, 0].set_yticklabels([])

            for spectrum_name in self.parser.names:
                if spectrum_name in header_name:
                    name = spectrum_name

            color = self.color_dict[name]

            start = self.parser.fit_start
            end = self.parser.fit_end

            handle = self.axs[row, 0].plot(x[start:end], y[start:end, j], c=color)

            self.axs[row, 0].set_xlim(left=np.max(x), right=np.min(x))

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

            self.axs[key, 0].legend(
                handles=handles_sort,
                labels=labels_sort,
                ncol=1,
                prop={"size": self.fontdict_legend["size"]},
                loc="best",
            )

        self.fig.tight_layout()

        return self.fig, self.axs


def main():
    """Plot of reference data with multiple elements."""
    datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"

    file_dict = {
        "filename": "nicofe_references.txt",
        "title": "",
        "fit_start": 0,
        "fit_end": -1,
    }

    wrapper = Wrapper(datafolder, file_dict)
    wrapper.parse_data()
    fig, ax = wrapper.plot_all()

    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(save_dir, "references_multiple" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
