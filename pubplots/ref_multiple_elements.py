# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:13:54 2022

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import TextParser, ParserWrapper

datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"

#%%
class FitTextParser(TextParser):
    """Parser for XPS data stored in TXT files."""

    def __init__(self):
        """
        Initialize empty data dictionary.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        super(FitTextParser, self).__init__()

    def _build_data(self, bg=True, envelope=True):
        """
        Build dictionary from the loaded data.

        Returns
        -------
        data_dict
            Data dictionary.

        """
        self.header_names = [hn.split(":")[1] for hn in self.header[6].split("\t") if "Cycle" in hn]

        if bg:
            self.header_names += ["Background"]
        if envelope:
            self.header_names += ["Envelope"]

        for i, hn in enumerate(self.header_names):
            if (hn == "Ni" or hn == "Co" or hn == "Fe"):
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
        c = [d.split() for d in self.data]

        x = lines[:, 0]
        y = lines[:, 1::2]

        y /= np.sum(y, axis=0)

        self.data = {"x": x, "y": y}

        return self.data

# %% lot of various different peak fitting methods

class MultipleElementWrapper(ParserWrapper):
    def __init__(self, datafolder, file_dict):
        super(MultipleElementWrapper, self).__init__(
            datafolder=datafolder, file_dict=file_dict
        )
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


    def parse_data(self, bg=True, envelope=True):
        filepath = os.path.join(self.datafolder, file_dict["filename"])
        self.parser = FitTextParser()
        self.parser.parse_file(filepath, bg=bg, envelope=bg)
        self.parser.title =file_dict["title"]
        self.parser.fit_start = file_dict["fit_start"]
        self.parser.fit_end = file_dict["fit_end"]

    def plot_all(self):
        self.fig, self.axs = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(12, 12),
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

            self.axs[row,0].set_xlabel(
                "Binding energy (eV)",
                fontdict=self.fontdict)
            self.axs[row,0].set_ylabel(
                "Intensity (arb. units)",
                fontdict=self.fontdict)
            self.axs[row,0].tick_params(
                axis="x",
                labelsize=self.fontdict["size"])
            self.axs[row,0].tick_params(
                axis="y",
                which="both",
                right=False,
                left=False)
            self.axs[row,0].set_yticklabels([])

            for spectrum_name in self.parser.names:
                if spectrum_name in header_name:
                    name = spectrum_name

            color = self.color_dict[name]

            start = self.parser.fit_start
            end = self.parser.fit_end

            handle = self.axs[row,0].plot(
                x[start:end], y[start:end, j], c=color
                )

            self.axs[row,0].set_xlim(left=np.max(x), right=np.min(x))

            if name not in handle_dict:
                handle_dict[name] = handle
                labels.append(name)

        handles = [x for l in list(handle_dict.values()) for x in l]
        labels = self._reformat_label_list(labels)

        # Only for this plot: Sort by oxidation state
        labels_sort_dict = {
            0: ["Ni metal","NiO"],
            1: ["Co metal", "CoO", "Co$_{3}$O$_{4}$"],
            2: ["Fe metal","FeO","Fe$_{3}$O$_{4}$", "Fe$_{2}$O$_{3}$"]
            }

        for key, label_list in labels_sort_dict.items():
            handles_sort = []
            labels_sort = []
            for label in label_list:
                index = labels.index(label)
                handles_sort.append(handles[index])
                labels_sort.append(label)

            self.axs[key,0].legend(
                handles=handles_sort,
                labels=labels_sort,
                ncol=1,
                prop={"size": self.fontdict_legend["size"]},
                loc="best"
                )

        self.fig.tight_layout()

        return self.fig, self.axs


# %% Plot of spectrum with multiple elements
file_dict = {
    "filename": "nicofe_references.txt",
    "title": "",
    "fit_start": 0,
    "fit_end": -1,
    }

wrapper = MultipleElementWrapper(datafolder, file_dict)
wrapper.parse_data(bg=False, envelope=False)
fig, ax = wrapper.plot_all()

plt.show()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures"
fig_path = os.path.join(save_dir, "ref_multiple_elements.png")
fig.savefig(fig_path)


