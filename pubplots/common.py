# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

#%%
def filter_header_list(header_list):
    header_list = list(filter(None, header_list))

    header_list = [hn for hn in header_list if hn != "\n" ]

    for i, hn in enumerate(header_list):
        if "\n" in hn:
            header_list[i] = hn.strip("\n")

    return header_list


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

        self.names = [
            "CPS",
            "Cu metal",
            "Cu2O",
            "CuO",
            "Co metal",
            "CoO",
            "Co3O4",
            "Fe metal",
            "FeO",
            "Fe3O4",
            "Fe2O3",
            "Ni metal",
            "NiO",
            "MnO",
            "Mn2O3",
            "MnO2",
            "Pd metal",
            "PdO",
            "Ti metal",
            "TiO",
            "Ti2O3",
            "TiO2",
            "Background",
            "Envelope",
        ]

    def parse_file(self, filepath, bg=True, envelope=True):
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
        return self._build_data(bg=bg, envelope=envelope)

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
        self.header = self.data[:7]
        self.data = self.data[7:]

    def _build_data(self, bg=True, envelope=True):
        """
        Build dictionary from the loaded data.

        Returns
        -------
        data_dict
            Data dictionary.

        """
        self.header_names = (
            ["CPS"]
            + self.header[0].split("\t")[2:]
        )

        if bg:
            self.header_names += ["Background"]
        if envelope:
            self.header_names += ["Envelope"]

        self.header_names = filter_header_list(self.header_names)

        lines = np.array([[float(i) for i in d.split()] for d in self.data])
        x = lines[:, 0]
        y = lines[:, 1:]

        y /= np.sum(y)

        self.data = {"x": x, "y": y}

        return self.data


class ParserWrapper:
    def __init__(self, datafolder, file_dict):
        self.datafolder = datafolder
        self.file_dict = file_dict

        self.parsers = []

        self.fontdict = {"size": 25}
        self.fontdict_legend = {"size": 16}
        self.color_dict = {
            "CPS": "grey",
            "Fe metal": "black",
            "FeO": "green",
            "Fe3O4": "crimson",
            "Fe2O3": "brown",
            "Fe sat": "deeppink",
            "Co metal": "darkturquoise",
            "CoO": "tab:purple",
            "Co3O4": "tab:olive",
            "Ni metal": "orange",
            "NiO": "navy",
            "Background": "tab:brown",
            "Envelope": "red",
        }

        self.label_dict = {
            "Cu2O": "Cu$_{2}$O",
            "Co3O4": "Co$_{3}$O$_{4}$",
            "Fe sat": "Fe satellite",
            "Fe3O4": "Fe$_{3}$O$_{4}$",
            "Fe2O3": "Fe$_{2}$O$_{3}$",
            "Fe sat": "Fe satellite",
            "Mn2O3": "Mn$_{2}$O$_{3}$",
            "MnO2": "MnO$_{2}$",
            "Ti2O3": "Ti$_{2}$O$_{3}$",
            "TiO2": "TiO$_{2}$",
        }

    def parse_data(self, bg=True, envelope=True):
        for method, d in self.file_dict.items():
            filepath = os.path.join(self.datafolder, d["filename"])
            parser = TextParser()
            parser.parse_file(filepath, bg=bg, envelope=envelope)
            parser.title = d["title"]
            parser.fit_start = d["fit_start"]
            parser.fit_end = d["fit_end"]
            parser.quantification = d["quantification"]
            self.parsers.append(parser)

    def _reformat_label_list(self, label_list):
        formatted_label_list = []

        for item in label_list:
            if item in self.label_dict:
                formatted_label_list.append(self.label_dict[item])
            else:
                formatted_label_list.append(item)

        return formatted_label_list

    def plot_all(
            self,
            with_fits=True,
            with_nn_col=True,
            with_quantification=True):

        width_ratios = [2]*len(self.parsers)
        ncols = len(self.parsers)

        if with_nn_col:
            width_ratios.append(1)
            ncols += 1

        self.fig, self.axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(len(self.parsers) * 12, 8),
            squeeze=False,
            dpi=300,
            gridspec_kw={"width_ratios": width_ratios},
        )

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            self.axs[0,i].set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            self.axs[0,i].set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
            self.axs[0,i].tick_params(axis="x", labelsize=self.fontdict["size"])
            self.axs[0,i].tick_params(
                axis="y",
                which="both",
                right=False,
                left=False)

            self.axs[0,i].set_yticklabels([])

            handle_dict = {}
            labels = []

            for j, header_name in enumerate(parser.header_names):
                for spectrum_name in parser.names:
                    if spectrum_name in header_name:
                        name = spectrum_name

                color = self.color_dict[name]

                if name == "CPS":
                    if not with_fits:
                        color = "black"
                    handle = self.axs[0,i].plot(x, y[:, j], c=color)
                else:
                    if with_fits:
                        start = parser.fit_start
                        end = parser.fit_end

                        try:
                            handle = self.axs[0,i].plot(
                                x[start:end], y[start:end, j], c=color
                                )
                        except IndexError:
                            handle = self.axs[0,i].plot(
                                x[start:end], y[start:end], c=color
                                )

                        if name not in handle_dict:
                            handle_dict[name] = handle
                            labels.append(name)

            handles = [x for l in list(handle_dict.values()) for x in l]
            labels = self._reformat_label_list(labels)

            #if not any(x in self.parsers[i].filepath for x in ("spectrum.txt")):
            if "spectrum.txt" not in self.parsers[i].filepath:
                self.axs[0,i].legend(
                    handles=handles,
                    labels=labels,
                    prop={"size": self.fontdict_legend["size"]},
                    loc="upper right",
                    )

            self.axs[0,i].set_title(parser.title, fontdict=self.fontdict)
            self.axs[0,i].set_xlim(left=np.max(x), right=np.min(x))
