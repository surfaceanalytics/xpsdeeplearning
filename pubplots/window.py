# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:51:00 2022

@author: pielsticker
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import csv

from common import TextParser, ParserWrapper

#%%

class WindowWrapper(ParserWrapper):
    """Parser for XPS data stored in TXT files."""

    def __init__(self, datafolder, file_dict):
        """
        Initialize empty data dictionary.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        super(WindowWrapper, self).__init__(
            datafolder=datafolder, file_dict=file_dict
            )
        self.fontdict_legend = {"size": 16}

    def load_history(self, csv_filepath):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing the previous training history.

        """
        history = {}
        with open(csv_filepath, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, item in row.items():
                    if key not in history.keys():
                        history[key] = []
                    history[key].append(float(item))

        self.history = history

        return self.history

    def _add_metric_plot(
        self,
        ax,
        history,
        metric,
        fontdict,
        title=None,
        ylabel=None
    ):
        """
        Plots the training and validation values of a metric
        against the epochs.

        Returns
        -------
        None.

        """
        fontdict = {"size": 25}
        fontdict_legend = {"size": 21}
        color_dict = {
            "train_loss": "cornflowerblue",
            "val_loss": "forestgreen",
        }

        metric_cap = metric.capitalize()
        legend = ["Training", "Validation"]

        metric_cap = metric.capitalize()
        if ylabel is None:
            ylabel = metric_cap
        ax.set_ylabel(str(ylabel), fontdict=fontdict)

        ax.set_title("(b) Training and validation loss", fontdict=fontdict)


        ax.set_xlabel("Epochs", fontdict=fontdict)
        ax.tick_params(axis="x", labelsize=fontdict["size"])
        ax.tick_params(axis="y", labelsize=fontdict["size"])


        ax.plot(self.history[metric], linewidth=3, c=color_dict["train_loss"])
        val_key = "val_" + metric
        ax.plot(self.history[val_key], linewidth=3,
                c=color_dict["val_loss"], alpha=0.6)
        ax.set_xlim(-5, len(self.history[metric]) + 5)

        ax.set_ylabel(str(ylabel), fontdict=fontdict)


        ax.legend(
            legend,
            title="MAE (L$_1$) loss",
            prop=fontdict_legend,
            title_fontsize=fontdict_legend["size"],
            bbox_to_anchor=(0, 0, 1, 1),
            fancybox=True,
            shadow=False,
            mode=None,
        )

        return ax



    def plot_all(self, with_fits=False):
        self.fig, self.ax = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(30, 8),
            squeeze=False,
            dpi=100,
            gridspec_kw={"width_ratios": [1.5, 1, 1.5]},
        )
        fontdict = {"size": 25}
        fontdict_small = {"size": 14}
        fontdict_legend = {"size": 17}
        color_dict = {
            "CPS": "black",
            "window": "orangered"
        }

        label_dict = {
            "Co3O4": "Co$_{3}$O$_{4}$",
            "Fe sat": "Fe satellite",
            "Fe3O4": "Fe$_{3}$O$_{4}$",
            "Fe2O3": "Fe$_{2}$O$_{3}$",
            "Fe sat": "Fe satellite",
        }
        col_labels = [
            "Co metal",
            "CoO",
            "Co$_{3}$O$_{4}$",
            "Fe metal",
            "FeO",
            "Fe$_{3}$O$_{4}$",
            "Fe$_{2}$O$_{3}$",
        ]

        self.ax[0,1] = self._add_metric_plot(
            ax = self.ax[0,1],
            history=history,
            metric="loss",
            fontdict=fontdict,
            title="MAE (L$_1$) loss",
            ylabel="MAE (L$_1$) loss",
        )

        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            if i == 1:
                i += 1

            self.ax[0,i].set_xlabel("Binding energy (eV)", fontdict=fontdict)
            self.ax[0,i].set_ylabel("Intensity (arb. units)", fontdict=fontdict)
            self.ax[0,i].tick_params(axis="x", labelsize=fontdict["size"])
            self.ax[0,i].set_yticklabels([])
            self.ax[0,i].tick_params(
                axis="y",
                which="both",
                right=False,
                left=False)

            if not with_fits:
                parser.header_names = ["CPS"]

            for j, header_name in enumerate(parser.header_names):
                for spectrum_name in parser.names:
                    if spectrum_name in header_name:
                        name = spectrum_name

                color = color_dict[name]

                start = parser.fit_start
                end = parser.fit_end

                try:
                    self.ax[0,i].plot(
                        x[start:end], y[start:end, j], c=color
                        )
                except IndexError:
                    self.ax[0,i].plot(
                        x[start:end], y[start:end], c=color
                        )
            if i == 0:
                text = "Ground truth:"
            else:
                text = "Quantification:"

            self.ax[0,i].set_title(parser.title, fontdict=fontdict)

            quantification = [
                f"{p} %" for p in list(parser.quantification.values())
            ]

            table = self.ax[0,i].table(
                cellText=[quantification],
                cellLoc="center",
                colLabels=col_labels,
                bbox=[0.03, 0.04, 0.095*len(quantification), 0.15],
                zorder=500
                )
            table.auto_set_font_size(False)
            table.set_fontsize(fontdict_small["size"])

            self.ax[0,i].text(
                0.03,
                0.22,
                text,
                horizontalalignment="left",
                size=fontdict_small["size"],
                verticalalignment="center",
                transform=self.ax[0,i].transAxes,
                )

        window = [705, 805]

        self.ax[0,0].set_xlim(left=np.max(x), right=np.min(x))
        self.ax[0,2].set_xlim(left=window[1], right=window[0])

        ymin, ymax = self.ax[0,0].get_ylim()

        rect = patches.Rectangle(
            (window[0], 0.27*ymax),
            window[1]-window[0],
            0.71*ymax,
            linewidth=2,
            edgecolor=color_dict["window"],
            facecolor="none",
            )
        self.ax[0,0].add_patch(rect)

        arrow = patches.FancyArrowPatch(
            (window[0], 0.65*ymax),
            (window[1],0.65*ymax),
            arrowstyle='<->',
            mutation_scale=20,
            color=color_dict["window"]
            )
        self.ax[0,0].add_patch(arrow)

        self.ax[0,0].text(
            (window[1]-window[0])/2+window[0],
            0.7*ymax,
            "100 eV window",
            horizontalalignment="center",
            size=fontdict["size"],
            color=color_dict["window"],
            verticalalignment="center",
            )

        self.fig.tight_layout()

        return self.fig, self.ax

#%% Plot of spectrum with multiple elements with window
datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
logpath = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs\20210914_19h13m_FeCo_combined_without_auger_7_classes_100eV_window\logs\log.csv"

file_dict = {
    "true": {
        "filename": "feco.txt",
        "title": "(a) Simulated spectrum with window",
        "fit_start": 0,
        "fit_end": -1,
        "shift_x": 0.0,
        "noise": 32.567,
        "FWHM": 1.28,
        "scatterer": "N2",
        "distance": 0.18,
        "pressure": 0.3,
        "quantification": {
            "Co metal": 26.7,
            "CoO": 22.17,
            "Co3O4": 24.09,
            "Fe metal": 0.0,
            "FeO": 14.37,
            "Fe3O4": 12.66,
            "Fe2O3": 0.0,
        },
    },
    "neural_net": {
        "filename": "feco.txt",
        "title": "(c) Neural network quantifcation on one example",
        "fit_start": 0,
        "fit_end": -1,
        "shift_x": 0.0,
        "quantification": {
            "Co metal": 25.73,
            "CoO": 22.84,
            "Co3O4":  28.26,
            "Fe metal": 4.12,
            "FeO": 10.53,
            "Fe3O4": 5.77,
            "Fe2O3": 2.75,
        },
    },
}


wrapper = WindowWrapper(datafolder, file_dict)
wrapper.parse_data(bg=True, envelope=True)
history = wrapper.load_history(logpath)

fig, ax = wrapper.plot_all()

texts = [
    ("Co 2p", 0.26, 0.9),
    ("Fe 2p", 0.79, 0.47),
    ]

for t in texts:
    ax[0,0].text(
        x=t[1],
        y=t[2],
        s=t[0],
        horizontalalignment="left",
        size=25,
        verticalalignment="center",
        transform=ax[0,0].transAxes
        )
plt.plot()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Identification & Quantification\figures"
fig_filename = f"window.png"
fig_path = os.path.join(save_dir, fig_filename)
fig.savefig(fig_path)
