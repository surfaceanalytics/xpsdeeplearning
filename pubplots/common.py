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
Common function for pubplots
"""
import os
from abc import ABC, abstractmethod
import numpy as np


REPO_PATH = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\xpsdeeplearning"
UTILS_FOLDER = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils"
DATAFOLDER = os.path.join(UTILS_FOLDER, "exports")
RUNFOLDER = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs"
SAVE_DIR = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\manuscript\figures"


def maximum_absolute_error(y_true, y_pred):
    """
    Get the highest absolute error between prediction and
    ground truth.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.max(np.abs(y_pred - y_true), axis=0)


def filter_header_list(header_list):
    """Filter a list of strings to only have non-nan values."""
    header_list = list(filter(None, header_list))

    header_list = [header_name for header_name in header_list if header_name != "\n"]

    for i, header_name in enumerate(header_list):
        if "\n" in header_name:
            header_list[i] = header_name.strip("\n")

    return header_list


def get_xlsxpath(fit_datafolder, method):
    """
    Get the path of a fit result csv.

    Parameters
    ----------
    fit_datafolder : str
        Path containing fit result csv files.
    method : str
        Peak fitting method.

    """
    return os.path.join(fit_datafolder, method + ".csv")


def sort_with_index(array, reverse=True):
    """
    Sort a numpy array by value and return a list with the
    indexes and values.

    Parameters
    ----------
    array : np.ndarray
        One dimensional numpy arra.
    reverse : bool, optional
        If True, the sorting is done in reverse.
        The default is True.

    Returns
    -------
    list
        List of strings with indices and values.

    """
    return [
        f"No. {j} : {k}"
        for (k, j) in sorted([(x, i) for (i, x) in enumerate(array)], reverse=reverse)
    ]


def print_mae_info(mae, name, precision=3):
    """Print information about the MAE for a given name."""
    print(name + ":")
    print(
        f"\t Mean MAE = {np.round(np.mean(mae), precision)}"
        + " \u00B1 "
        + f"{np.round(np.std(mae), precision)} \n",
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
        self.filepath: str = None
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
        self.header = []
        self.header_names = []
        self.data = []

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
        with open(filepath) as file:
            for line in file:
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
        header_names = ["CPS"] + self.header[0].split("\t")[2:]

        if "background" in kwargs.keys():
            header_names += ["Background"]
        if "envelope" in kwargs.keys():
            header_names += ["Envelope"]

        self.header_names = filter_header_list(header_names)

        lines = np.array([[float(i) for i in d.split()] for d in self.data])
        x = lines[:, 0]
        y = lines[:, 1:]

        y /= np.sum(y)

        self.data = {"x": x, "y": y}

        return self.data


@abstractmethod
class ParserWrapper(ABC):
    """Abstract wrapper for loading and plotting."""

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
            "Fe3O4": "Fe$_{3}$O$_{4}$",
            "Fe2O3": "Fe$_{2}$O$_{3}$",
            "Fe sat": "Fe satellite",
            "Mn2O3": "Mn$_{2}$O$_{3}$",
            "MnO2": "MnO$_{2}$",
            "Ti2O3": "Ti$_{2}$O$_{3}$",
            "TiO2": "TiO$_{2}$",
        }

    def parse_data(self, background=True, envelope=True):
        """Parse data from file dict."""
        for single_dict in self.file_dict.values():
            filepath = os.path.join(self.datafolder, single_dict["filename"])
            parser = TextParser()
            parser.parse_file(filepath, background=background, envelope=envelope)
            for key, value in single_dict.items():
                setattr(parser, key, value)
            self.parsers.append(parser)

    def _reformat_label_list(self, label_list):
        """Reformat lavbel list according to label dict."""
        formatted_label_list = []

        for item in label_list:
            if item in self.label_dict:
                formatted_label_list.append(self.label_dict[item])
            else:
                formatted_label_list.append(item)

        return formatted_label_list

    @abstractmethod
    def plot_all(self, with_fits=True, with_nn_col=True, with_quantification=True):
        """Abstract method for plotting the results."""
