# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:13:54 2022

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from common import TextParser, ParserWrapper

datafolder = (
    r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
)

#%%
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
            hn.split(":")[1]
            for hn in self.header[6].split("\t")
            if "Cycle" in hn
        ]

        if bg:
            self.header_names += ["Background"]
        if envelope:
            self.header_names += ["Envelope"]

        for i, hn in enumerate(self.header_names):
            if hn == "Ni" or hn == "Co" or hn == "Fe":
                self.header_names[i] += " metal"

        lines = np.array([[float(i) for i in d.split()] for d in self.data])

        x = lines[:, 0::2]
        y = lines[:, 1::2]

        y /= np.sum(y, axis=0)

        self.data = {"x": x, "y": y}

        return self.data


# %%
class Wrapper(ParserWrapper):
    def __init__(self, datafolder, file_dict):
        super(Wrapper, self).__init__(
            datafolder=datafolder, file_dict=file_dict
        )
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

    def parse_data(self, bg=True, envelope=True):
        for element, d in self.file_dict.items():
            filepath = os.path.join(self.datafolder, d["filename"])
            parser = FitTextParser()
            parser.parse_file(filepath, bg=bg, envelope=envelope)
            parser.title = d["title"]
            parser.fit_start = d["fit_start"]
            parser.fit_end = d["fit_end"]
            self.parsers.append(parser)

    def plot_all(self):
        ncols = 2

        self.fig = plt.figure(figsize=(18, 24), dpi=300)
        gs = gridspec.GridSpec(4, ncols * 2, wspace=0.5, hspace=0.5)

        # Set up 2 plots in first three row and 1 in last row.
        ax0_0 = self.fig.add_subplot(gs[0, :2])
        ax0_1 = self.fig.add_subplot(gs[0, 2:])
        ax1_0 = self.fig.add_subplot(gs[1, :2])
        ax1_1 = self.fig.add_subplot(gs[1, 2:])
        ax2_0 = self.fig.add_subplot(gs[2, :2])
        ax2_1 = self.fig.add_subplot(gs[2, 2:])
        ax3 = self.fig.add_subplot(gs[3, 1:3])

        self.axs = np.array(
            [[ax0_0, ax0_1], [ax1_0, ax1_1], [ax2_0, ax2_1], [ax3, None]]
        )

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            row, col = int(i / ncols), i % ncols

            self.axs[row, col].set_xlabel(
                "Binding energy (eV)", fontdict=self.fontdict
            )
            self.axs[row, col].set_ylabel(
                "Intensity (arb. units)", fontdict=self.fontdict
            )
            self.axs[row, col].tick_params(
                axis="x", labelsize=self.fontdict["size"]
            )
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

                handle = self.axs[row, col].plot(
                    x[start:end, j], y[start:end, j], c=color, linewidth=2
                )

                if name not in handle_dict:
                    handle_dict[name] = handle
                    labels.append(name)

                handles = [x for l in list(handle_dict.values()) for x in l]
                labels = self._reformat_label_list(labels)

            self.axs[row, col].legend(
                handles=handles,
                labels=labels,
                ncol=1,
                prop={"size": self.fontdict_legend["size"]},
                loc="upper left",
            )

            self.axs[row, col].set_title(parser.title, fontdict=self.fontdict)
            self.axs[row, col].set_xlim(left=x[start, j], right=x[end, j])

        gs.tight_layout(self.fig)

        return self.fig, self.axs


# %% Plot of spectrum with multiple elements
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

wrapper = Wrapper(datafolder, file_dict)
wrapper.parse_data(bg=False, envelope=False)
fig, ax = wrapper.plot_all()

plt.show()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\figures"
fig_path = os.path.join(save_dir, "references_single.png")
fig.savefig(fig_path)
