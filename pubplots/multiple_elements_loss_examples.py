# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:50:14 2021

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import csv

from common import TextParser, ParserWrapper

datafolder = (
    r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\utils\exports"
)

# %%
class Wrapper(ParserWrapper):
    def __init__(self, datafolder, file_dict):
        super(Wrapper, self).__init__(
            datafolder=datafolder, file_dict=file_dict
        )
        self.fontdict = {"size": 38}
        self.fontdict_inset = {"size": 30}
        self.fontdict_small = {"size": 20}
        self.fontdict_mae = {"size": 28}

    def parse_data(self, bg=True, envelope=True):
        for result_dict in self.file_dict.values():
            for method, d in result_dict.items():
                filepath = os.path.join(self.datafolder, d["filename"])
                parser = TextParser()
                parser.parse_file(filepath, bg=bg, envelope=envelope)
                for key, value in d.items():
                    setattr(parser, key, value)
                self.parsers.append(parser)

    def get_total_history(self, clf_name):
        """
        Load the previous training history from the CSV log file.

        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing the previous training history.

        """
        input_datafolder = (
            r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\runs"
        )
        csv_filepath = os.path.join(
            *[input_datafolder, clf_name, "logs/log.csv"]
        )

        self.history = {}
        try:
            with open(csv_filepath, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    for key, item in row.items():
                        if key not in self.history.keys():
                            self.history[key] = []
                        self.history[key].append(float(item))
        except FileNotFoundError:
            pass

        return self.history

    def _plot_metric(
        self,
        ax,
        history,
        metric,
        title=None,
        ylabel=None,
        zoom=False,
        zoom_x=(None, None),
        zoom_y=(None, None),
        to_file=False,
    ):
        metric_cap = metric.capitalize()

        if ylabel is None:
            ylabel = metric_cap
        ax.set_ylabel(str(ylabel), fontdict=self.fontdict)
        ax.set_title(title, fontdict=self.fontdict)

        ax.set_xlabel("Epochs", fontdict=self.fontdict)
        ax.tick_params(axis="x", labelsize=self.fontdict["size"])
        ax.tick_params(axis="y", labelsize=self.fontdict["size"])

        colors = ["cornflowerblue", "red", "forestgreen", "darkgrey"]

        metric_history = history[metric]
        (h0,) = ax.plot(metric_history, linewidth=3, c=colors[0])
        val_key = "val_" + metric
        (h1,) = ax.plot(history[val_key], linewidth=3, c=colors[1], alpha=0.6)
        ax.set_xlim(-5, len(metric_history) + 5)

        labels = ["Training", "Validation"]

        ax.legend(
            handles=[h0, h1],
            labels=labels,
            prop=self.fontdict,
            title="MAE (L$_1$) loss",
            title_fontsize=self.fontdict["size"],
            bbox_to_anchor=(0, 0, 1, 1),
            fancybox=True,
            shadow=False,
            mode=None,
        )

        if zoom:
            axins = inset_axes(
                ax,
                width=6,
                height=4.5,
                bbox_to_anchor=(0.325, 0.65),
                bbox_transform=ax.figure.transFigure,
            )
            axins.set_xlabel("Epochs", fontdict=self.fontdict_inset)
            axins.set_ylabel(str(ylabel), fontdict=self.fontdict_inset)
            axins.tick_params(axis="x", labelsize=self.fontdict_inset["size"])
            axins.tick_params(axis="y", labelsize=self.fontdict_inset["size"])
            axins.plot(metric_history, linewidth=3, c=colors[0])
            axins.plot(history[val_key], linewidth=3, c=colors[1], alpha=0.5)
            if zoom_x[1] is None:
                zoom_end = len(metric_history) + 5
            else:
                zoom_end = zoom_x[1]
            axins.set_xlim(zoom_x[0], zoom_end)  # apply the x-limits
            axins.set_ylim(zoom_y[0], zoom_y[1])

        return ax

    def _add_spectrum_examples(self, ax, parser, color):
        x, y = parser.data["x"], parser.data["y"]
        start, end = parser.fit_start, parser.fit_end

        ax.set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
        ax.set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
        ax.tick_params(axis="x", labelsize=self.fontdict["size"])
        ax.tick_params(axis="y", which="both", right=False, left=False)

        ax.set_yticklabels([])

        for j, header_name in enumerate(parser.header_names):
            if header_name == "CPS":
                handle = ax.plot(x, y[:, j], c=color)
            else:
                start = parser.fit_start
                end = parser.fit_end
                try:
                    handle = ax.plot(x[start:end], y[start:end, j], c=color)
                except IndexError:
                    handle = ax.plot(x[start:end], y[start:end], c=color)

            ax.set_title(parser.title, fontdict=self.fontdict)
            ax.set_xlim(left=np.max(x), right=np.min(x))
            ax.set_ylim(bottom=np.min(y)*0.8)

            percentages = [
                [f"{p[0]} %", f"{p[1]} %"]
                for p in parser.quantification.values()
            ]

            keys = list(parser.quantification.keys())
            row_labels = self._reformat_label_list(keys)

            table = ax.table(
                cellText=percentages,
                cellLoc="center",
                colLabels=["Truth", "CNN"],
                rowLabels=row_labels,
                bbox=[0.15, 0.025, 0.2, 0.45],
                zorder=500,
            )
            table.set_fontsize(self.fontdict_small["size"])

            mae = f"MAE:\n{np.round(parser.mae,3)}"
            ax.text(
                x=0.85,
                y=0.85,
                s=mae,
                size=self.fontdict_mae["size"],
                verticalalignment="center",
                transform=ax.transAxes,
            )

        return ax

    def plot_all(self):
        nrows = 2
        ncols = 3

        self.fig = plt.figure(figsize=(32, 15), dpi=300)
        gs = gridspec.GridSpec(
            nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.3
        )

        ax0 = self.fig.add_subplot(gs[:, 0])
        ax0_1 = self.fig.add_subplot(gs[0, 1])
        ax0_2 = self.fig.add_subplot(gs[0, 2])
        ax1_1 = self.fig.add_subplot(gs[1, 1])
        ax1_2 = self.fig.add_subplot(gs[1, 2])

        self.axs = [
            ax0,
            [[ax0_1, ax0_2], [ax1_1, ax1_2]],
        ]

        ax0 = self._plot_metric(
            ax0,
            history=history,
            metric="loss",
            title="(a) Training and validation loss",
            ylabel="MAE (L$_1$) loss",
            zoom=True,
            zoom_x=[100, None],
            zoom_y=[None, 0.04],
        )

        ax0_1 = self._add_spectrum_examples(
            ax0_1, self.parsers[0], color="green"
        )
        ax0_2 = self._add_spectrum_examples(
            ax0_2, self.parsers[1], color="green"
        )
        ax1_1 = self._add_spectrum_examples(
            ax1_1, self.parsers[2], color="red"
        )
        ax1_2 = self._add_spectrum_examples(
            ax1_2, self.parsers[3], color="red"
        )

        gs.tight_layout(self.fig)

        return self.fig, self.axs


# %%
file_dict = {
    "sucesses": {
        "NiCoFe_good_0": {
            "filename": "nicofe_good_fit_no_81.txt",
            "title": "(b)",
            "fit_start": 0,
            "fit_end": -1,
            "mae": 0.00833,
            "quantification": {
                "Ni metal": [0.0, 0.0],
                "NiO": [35.9, 35.1],
                "Co metal": [29.0, 26.1],
                "CoO": [0.0, 0.0],
                "Co3O4": [35.1, 38.8],
                "Fe metal": [0.0, 0.0],
                "FeO": [0.0, 0.0],
                "Fe3O4": [0.0, 0.0],
                "Fe2O3": [0.0, 0.0],
            },
        },
        "NiCoFe_good_1": {
            "filename": "nicofe_good_fit_no_45.txt",
            "title": "(c)",
            "fit_start": 0,
            "fit_end": -1,
            "mae": 0.0132,
            "quantification": {
                "Ni metal": [25.9, 21.3],
                "NiO": [19.1, 20.5],
                "Co metal": [15.0, 15.9],
                "CoO": [20.1, 23.7],
                "Co3O4": [0.0, 0.0],
                "Fe metal": [19.9, 18.6],
                "FeO": [0.0, 0.0],
                "Fe3O4": [0.0, 0.0],
                "Fe2O3": [0.0, 0.0],
            },
        },
    },
    "fails": {
        "NiCoFe_bad_0": {
            "filename": "nicofe_bad_fit_no_71.txt",
            "title": "(d)",
            "fit_start": 0,
            "fit_end": -1,
            "mae": 0.06984,
            "quantification": {
                "Ni metal": [0.0, 0.0],
                "NiO": [32.6, 22.5],
                "Co metal": [0.0, 0.0],
                "CoO": [0.0, 0.0],
                "Co3O4": [0.0, 0.0],
                "Fe metal": [0.0, 0.0],
                "FeO": [0.0, 12.9],
                "Fe3O4": [28.2, 6.9],
                "Fe2O3": [39.2, 57.7],
            },
        },
        "NiCoFe_bad_1": {
            "filename": "nicofe_bad_fit_no_18.txt",
            "title": "(e)",
            "fit_start": 0,
            "fit_end": -1,
            "mae": 0.05695,
            "quantification": {
                "Ni metal": [20.5, 12.2],
                "NiO": [20.7, 16.9],
                "Co metal": [0.0, 0.0],
                "CoO": [0.0, 0.0],
                "Co3O4": [0.0, 0.0],
                "Fe metal": [22.5, 16.6],
                "FeO": [0.0, 11.2],
                "Fe3O4": [21.5, 13.9],
                "Fe2O3": [14.8, 29.2],
            },
        },
    },
}

wrapper = Wrapper(datafolder, file_dict)
wrapper.parse_data(bg=False, envelope=False)

classifier = (
    "20210604_23h09m_NiCoFe_9_classes_long_linear_comb_small_gas_phase"
)
history = wrapper.get_total_history(classifier)

fig, axs = wrapper.plot_all()

save_dir = r"C:\Users\pielsticker\Lukas\MPI-CEC\Publications\DeepXPS paper\Manuscript - Automatic Quantification\figures"
fig_path = os.path.join(save_dir, "loss_examples_multiple.png")
fig.savefig(fig_path, bbox_inches="tight")
