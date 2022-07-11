# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:23:14 2022

@author: pielsticker
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:12:28 2021

@author: pielsticker
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:15:57 2020

@author: Mark
"""
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
class TextParser:
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
        self.data_dict = []

    def parse_file(self, filepath):
        """
        Parse the .txt file into a list of dictionaries.

        Each dictionary is a grouping of related attributes.
        These are later put into a heirarchical nested dictionary that
        represents the native data structure of the export, and is well
        represented by JSON.
        """
        self.filepath = filepath
        self._read_lines(filepath)
        self._parse_header()
        return self._build_data()

    def _read_lines(self, filepath):
        self.data = []
        self.filepath = filepath
        with open(filepath) as fp:
            for line in fp:
                self.data += [line]

    def _parse_header(self):
        """
        Separate the dataset into header and data.

        Returns
        -------
        None.

        """
        self.header = self.data[:7]  # str(self.data[0]).split("\n")[0]
        self.data = self.data[7:]

    def _build_data(self):
        """
        Build dictionary from the loaded data.

        Returns
        -------
        data_dict
            Data dictionary.

        """
        self.header_names = (
            ["CPS"]
            + self.header[0].split("\t")[2:][:-4]
            + ["Background", "Envelope"]
        )

        self.names = [
            "CPS",
            "Fe metal",
            "FeO",
            "Fe3O4",
            "Fe2O3",
            "Fe sat",
            "Background",
            "Envelope",
        ]
        lines = np.array([[float(i) for i in d.split()] for d in self.data])
        x = lines[:, 0]
        y = lines[:, 1:]

        self.data = {"x": x, "y": y}

        return self.data


class ParserWrapper:
    def __init__(self, datafolder, file_dict):
        self.file_dict = file_dict
        self.parsers = []

        for method, d in file_dict.items():
            filepath = os.path.join(datafolder, d["filename"])
            parser = TextParser()
            parser.parse_file(filepath)
            parser.title = d["title"]
            parser.fit_start = d["fit_start"]
            parser.fit_end = d["fit_end"]
            parser.quantification = d["quantification"]
            self.parsers.append(parser)

    def plot_all(
            self,
            with_nn_col=True,
            with_quantification=True,
            to_file=False):

        width_ratios = [2]*len(self.parsers)
        ncols = len(self.parsers)

        if with_nn_col:
            width_ratios.append(1)
            ncols += 1

        fig, ax = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(len(self.parsers) * 12, 8),
            squeeze=False,
            dpi=300,
            gridspec_kw={"width_ratios": width_ratios},
        )
        fontdict = {"size": 25}
        fontdict_small = {"size": 21}
        fontdict_legend = {"size": 17}
        color_dict = {
            "CPS": "grey",
            "Fe metal": "black",
            "FeO": "green",
            "Fe3O4": "blue",
            "Fe2O3": "orange",
            "Fe sat": "deeppink",
            "Background": "tab:brown",
            "Envelope": "red",
        }

        label_dict = {
            "Fe sat": "Fe satellite",
            "Fe3O4": "Fe$_{3}$O$_{4}$",
            "Fe2O3": "Fe$_{2}$O$_{3}$",
            "Fe sat": "Fe satellite",
        }
        col_labels = ["Fe metal", "FeO", "Fe$_{3}$O$_{4}$", "Fe$_{2}$O$_{3}$"]

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            ax[0,i].set_xlabel("Binding energy (eV)", fontdict=fontdict)
            ax[0,i].set_ylabel("Intensity (arb. units)", fontdict=fontdict)
            ax[0,i].tick_params(axis="x", labelsize=fontdict["size"])
            ax[0,i].set_yticklabels([])

            handle_dict = {}
            labels = []

            for j, header_name in enumerate(parser.header_names):
                for spectrum_name in parser.names:
                    if spectrum_name in header_name:
                        name = spectrum_name

                    color = color_dict[name]
                if name == "CPS":
                    handle = ax[0,i].plot(x, y[:, j], c=color)
                else:
                    start = parser.fit_start
                    end = -parser.fit_end
                    try:
                        handle = ax[0,i].plot(
                            x[start:end], y[start:end, j], c=color
                        )
                    except IndexError:
                        handle = ax[0,i].plot(
                            x[start:end], y[start:end], c=color
                        )

                    if name not in handle_dict:
                        handle_dict[name] = handle
                        if name in label_dict:
                            labels.append(label_dict[name])
                        else:
                            labels.append(name)

            handles = [x for l in list(handle_dict.values()) for x in l]

            if "spectrum.txt" not in self.parsers[i].filepath:
                ax[0,i].legend(
                    handles=handles,
                    labels=labels,
                    prop={"size": fontdict_legend["size"]},
                    loc="upper left",
                )
                text = "Quantification:"
            else:
                text = "Ground truth:"

            ax[0,i].set_title(parser.title, fontdict=fontdict)

            quantification = [
                f"{p} %" for p in list(parser.quantification.values())
            ]

            if with_quantification:
                table = ax[0,i].table(
                    cellText=[quantification],
                    cellLoc="center",
                    colLabels=col_labels,
                    bbox=[0.03, 0.03, 0.6, 0.15],
                    )
                table.auto_set_font_size(False)
                table.set_fontsize(fontdict_small["size"])

                ax[0,i].text(
                    0.03,
                    0.215,
                    text,
                    horizontalalignment="left",
                    size=fontdict_small["size"],
                    verticalalignment="center",
                    transform=ax[0,i].transAxes,
                    )

            ax[0,i].set_xlim(left=np.max(x), right=np.min(x))

        if with_nn_col:
            ax[0,-1].set_axis_off()
            ax[0,-1].set_title("(e) Neural Network", fontdict=fontdict)

        fig.tight_layout()
        plt.show()

        if to_file:
            fig_name = os.path.join(datafolder, f"peak_fitting.png")
            fig.savefig(fig_name)  # , bbox_inches="tight")

        return fig


# %%
datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
file_dict = {
    "true": {
        "filename": "spectrum.txt",
        "title": "(a) Simulated spectrum",
        "fit_start": 0,
        "fit_end": 0,
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
        "fit_end": 137,
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
        "fit_end": 145,
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
        "fit_end": 137,
        "quantification": {
            "Fe metal": 38.0,
            "FeO": 11.9,
            "Fe3O4": 28.3,
            "Fe2O3": 21.7,
        },
    },
}

wrapper = ParserWrapper(datafolder, file_dict)
wrapper.plot_all(
    with_nn_col=True,
    with_quantification=True,
    to_file=True)
