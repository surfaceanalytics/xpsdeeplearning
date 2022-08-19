# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

#%%
def filter_header_list(header_list):
    header_list = list(filter(None, header_list))

    header_list = [hn for hn in header_list if hn != "\n" ]

    for i, hn in enumerate(header_list):
        if "\n" in hn:
            header_list[i] = hn.strip("\n")

    return header_list

def get_xlsxpath(fit_datafolder, method):
    return os.path.join(fit_datafolder, method + ".csv")

def sort_with_index(array, reverse=True):
    return [f"No. {j} : {k}" for (k, j) in sorted([(x, i) for (i, x) in enumerate(array)],reverse=reverse)]

def print_mae_info(mae, name, precision=3):
    print(name + ":")
    print(
        f"\t Mean MAE = {np.round(np.mean(mae), precision)}" + " \u00B1 " +  f"{np.round(np.std(mae), precision)} \n",
        f"\t Median MAE = {np.round(np.median(mae), precision)} \n",
        f"\t 25th percentile = {np.round(np.percentile(mae, 25), precision)} \n"
        f"\t 75th percentile = {np.round(np.percentile(mae, 75), precision)} \n",
        f"\t Maximum MAE = {np.round(np.max(mae), precision)}, Spectrum no. {np.argmax(mae)} \n",
        f"\t Minimum MAE = {np.round(np.min(mae), precision)}, Spectrum no. {np.argmin(mae)} \n",
        f"\t Lowest 5 MAEs = {sort_with_index(np.round(mae, precision), reverse=False)[:5]} \n",
        f"\t Highest 5 MAEs = {sort_with_index(np.round(mae, precision), reverse=True)[:5]} \n",
        )


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
            "Fe sat",
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

    def parse_file(self, filepath, **kwargs):
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
        return self._build_data(**kwargs)

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

    def _build_data(self, **kwargs):
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

        if "bg" in kwargs.keys():
            self.header_names += ["Background"]
        if "envelope" in kwargs.keys():
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
        self.fontdict_small = {"size": 20}
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
            for key, value in d.items():
                setattr(parser, key, value)
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
        pass
