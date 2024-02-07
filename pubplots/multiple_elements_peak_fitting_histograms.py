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
Peak fitting histograms for spectra with more than one element.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

from common import (
    ParserWrapper,
    get_xlsxpath,
    print_mae_info,
    UTILS_FOLDER,
    DATAFOLDER,
    SAVE_DIR,
)


class Wrapper(ParserWrapper):
    def __init__(self, datafolder, file_dict):
        super().__init__(datafolder=datafolder, file_dict=file_dict)
        self.fontdict_small = {"size": 13}
        self.fontdict_legend = {"size": 14.5}
        self.losses_test = {}

    def _add_loss_histogram(
        self,
        ax,
        losses_test,
    ):
        """
        Plots a histogram of the lossses for each example in the pred_test
        data set.

        Returns
        -------
        None.

        """

        ax.set_title("(b) MAEs on test data set", fontdict=self.fontdict)

        ax.set_xlabel("Mean Absolute Error ", fontdict=self.fontdict)
        ax.set_ylabel("Counts", fontdict=self.fontdict)
        ax.tick_params(axis="x", labelsize=self.fontdict["size"])
        ax.tick_params(axis="y", labelsize=self.fontdict["size"])

        N, bins, hist_patches = ax.hist(
            losses_test,
            bins=100,
            histtype="bar",
            fill=True,
            cumulative=False,
        )

        ax.set_xlim(
            hist_patches[0].xy[0],
            hist_patches[-1].xy[0],
        )
        ax.tick_params(axis="x", pad=12)

        return ax

    def _add_peak_fits(self, ax, with_fits=True):
        for i, parser in enumerate(self.parsers):
            x, y = parser.data["x"], parser.data["y"]

            ax.set_xlabel("Binding energy (eV)", fontdict=self.fontdict)
            ax.set_ylabel("Intensity (arb. units)", fontdict=self.fontdict)
            ax.tick_params(axis="x", labelsize=self.fontdict["size"])
            ax.tick_params(axis="y", which="both", right=False, left=False)
            ax.set_yticklabels([])

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
                    handle = ax.plot(x, y[:, j], c=color)
                else:
                    if with_fits:
                        start = parser.fit_start
                        end = parser.fit_end

                        try:
                            handle = ax.plot(
                                x[start:end],
                                y[start:end, j],
                                c=color,
                                linewidth=2,
                            )
                        except IndexError:
                            handle = ax.plot(x[start:end], y[start:end], c=color)

                        if name not in handle_dict:
                            handle_dict[name] = handle
                            labels.append(name)

            handles = [
                x for handle_list in list(handle_dict.values()) for x in handle_list
            ]
            labels = self._reformat_label_list(labels)

            if with_fits:
                ax.legend(
                    handles=handles,
                    labels=labels,
                    prop={"size": self.fontdict_legend["size"]},
                    loc="upper right",
                )

            ax.set_title(parser.title, fontdict=self.fontdict)
            ax.set_xlim(left=np.max(x[start:end]), right=np.min(x[start:end]))

        texts = [
            ("Ni 2p", 0.065, 0.65),
            ("Co 2p", 0.45, 0.575),
            ("Fe 2p", 0.85, 0.4),
        ]

        for t in texts:
            self.axs[0, i].text(
                x=t[1],
                y=t[2],
                s=t[0],
                horizontalalignment="left",
                size=25,
                verticalalignment="center",
                transform=self.axs[0, i].transAxes,
            )

        for i, parser in enumerate(self.parsers):
            self.axs[0, i].set_title(parser.title, fontdict=self.fontdict)

            q_true = [f"{p[0]} %" for p in list(parser.quantification.values())]
            q_fitting = [f"{p[1]} %" for p in list(parser.quantification.values())]
            q_cnn = [f"{p[2]} %" for p in list(parser.quantification.values())]

            keys = list(parser.quantification.keys())
            col_labels = self._reformat_label_list(keys)

            table = self.axs[0, i].table(
                cellText=[q_true, q_fitting, q_cnn],
                cellLoc="center",
                colLabels=col_labels,
                rowLabels=["Truth", "Peak fitting", "CNN"],
                bbox=[0.12, 0.79, 0.69, 0.15],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.fontdict_small["size"])

            self.axs[0, i].text(
                0.01,
                0.96,
                "Quantification:",
                horizontalalignment="left",
                size=self.fontdict_small["size"],
                verticalalignment="center",
                transform=self.axs[0, i].transAxes,
            )

        ax.set_ylim(top=np.max(parser.data["y"] * 1.1))

    def plot_all(self, with_fits=True):
        self.fig, self.axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(len(self.parsers) * 19, 12),
            squeeze=False,
            dpi=300,
            gridspec_kw={"width_ratios": [2, 1]},
        )

        self.axs[0, 0] = self._add_peak_fits(self.axs[0, 0], with_fits=with_fits)

        loss_legend = []
        for method, losses_test in self.losses_test.items():
            self.axs[0, 1] = self._add_loss_histogram(
                self.axs[0, 1],
                losses_test,
            )
            loss_legend.append(method)

        self.axs[0, 1].legend(
            labels=loss_legend,
            prop=self.fontdict_legend,
            bbox_to_anchor=(0, 0, 1, 1),
            fancybox=True,
            shadow=False,
            mode=None,
        )

        self.fig.tight_layout(w_pad=3.0)

        return self.fig, self.axs


def main():
    """Peak fitting histograms for spectra with more than one element."""
    file_dict = {
        "nicofe": {
            "filename": "nicofe_fit.txt",
            "title": "(a)",
            "fit_start": 15,
            "fit_end": -15,
            "shift_x": 0.0,
            "noise": 0.0,
            "FWHM": 0.0,
            "scatterer": None,
            "distance": 0.0,
            "pressure": 0.0,
            "quantification": {
                "Ni metal": [25.9, 20.4, 21.3],
                "NiO": [19.1, 18.5, 20.5],
                "Co metal": [15.0, 13.1, 15.9],
                "CoO": [20.1, 30.1, 23.7],
                "Co3O4": [0.0, 2.2, 0.0],
                "Fe metal": [19.9, 15.0, 18.6],
                "FeO": [0.0, 0.1, 0.0],
                "Fe3O4": [0.0, 0.0, 0.0],
                "Fe2O3": [0.0, 0.6, 0.0],
            },
        }
    }

    wrapper = Wrapper(DATAFOLDER, file_dict)
    wrapper.parse_data(bg=True, envelope=True)

    fit_datafolder = os.path.join(UTILS_FOLDER, "fit_comparison")
    cols = [
        "Ni metal",
        "NiO",
        "Co metal",
        "CoO",
        "Co3O4",
        "Fe metal",
        "FeO",
        "Fe3O4",
        "Fe2O3",
    ]
    df_true = pd.read_csv(
        get_xlsxpath(fit_datafolder, "truth_multiple"), index_col=0, sep=";"
    )
    df_true = df_true[cols]

    df_fit = pd.read_csv(
        get_xlsxpath(fit_datafolder, "lineshapes_multiple"), index_col=0, sep=";"
    )
    df_fit = df_fit.divide(100)
    df_fit.columns = df_fit.columns.str.replace(" %", "")
    df_fit = df_fit[cols]
    mae_fit = mean_absolute_error(
        df_fit.to_numpy().T, df_true.to_numpy().T, multioutput="raw_values"
    )
    wrapper.losses_test["Peak fit"] = mae_fit
    print_mae_info(mae_fit, "Lineshapes")

    df_nn = pd.read_csv(
        get_xlsxpath(fit_datafolder, "nn_multiple"), index_col=0, sep=";"
    )
    df_nn = df_nn[cols]
    mae_nn = mean_absolute_error(
        df_nn.to_numpy().T, df_true.to_numpy().T, multioutput="raw_values"
    )
    wrapper.losses_test["Neural network"] = mae_nn
    print_mae_info(mae_nn, "Neural network")

    fig, ax = wrapper.plot_all(with_fits=True)
    plt.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "fit_histograms_multiple" + ext)
        fig.savefig(fig_path, bbox_inches="tight")

    dfs = {"Neural network": df_nn, "Lineshapes": df_fit}

    def print_diff(df, threshold, kind="average"):
        df_test = df.copy()
        correct = 0
        wrong = 0
        for index in df.index:
            df_print = pd.concat(
                [df_true.loc[index], df_test.loc[index]], axis=1
            )  # .reset_index()
            df_print.columns = ["true", "nn"]
            df_print["diff"] = (df_true.loc[index] - df_test.loc[index]) * 100
            # print(f"{index}: max diff.: {max(df_print["diff"])}, mean diff.: {np.mean(np.abs(df_print["diff"]))}")
            if kind == "average":
                metric = np.mean(np.abs(df_print["diff"]))
            elif kind == "max":
                metric = np.max(df_print["diff"])
            if metric < threshold:
                correct += 1
            else:
                wrong += 1
        print(
            f"No. of correct classifications ({kind}): {correct}/{len(df_nn.index)} = {correct/len(df_nn.index)*100} %"
        )
        print(
            f"No. of wrong classifications ({kind}): {wrong}/{len(df_nn.index)} = {wrong/len(df_nn.index)*100} %"
        )

    threshold = 10.0  # percent
    for method, df in dfs.items():
        print(f"{method}:")
        print_diff(df, threshold, kind="average")
        print_diff(df, threshold, kind="max")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
