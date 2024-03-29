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
Plot of data class distribution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import SAVE_DIR

# noqa: E402
from xpsdeeplearning.network.data_handling import DataHandler
from xpsdeeplearning.network.utils import ClassDistribution

plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


class UpdatedClassDistribution(ClassDistribution):
    """ClassDistribution with enhanced plot functionality."""

    def plot(self, labels):
        """
        Plot the class distribution. Using the labels list as legend.

        Parameters
        ----------
        labels : list
            List of label values for the legend.

        Returns
        -------
        None.

        """
        fontdict = {"size": 14}
        fontdict_small = {"size": 12}
        colors = ("black", "green", "blue", "orange")

        fig = plt.figure(dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        x = np.arange(len(self.cd.keys())) * 1.5
        data = []

        if self.task == "regression":
            # ax.set_title("Average distribution across the classes", fontdict=fontdict)
            # Plot of the average label distribution in the different
            # data sets.
            for value in self.cd.values():
                data.append(value)
            data = np.transpose(np.array(data))
            ax.set_ylabel("Average concentration", fontdict=fontdict)

        else:
            ax.set_title("Class distribution", fontdict=fontdict)
            for value_dict in self.cd.values():
                data_list = []
                for value in value_dict.values():
                    data_list.append(value)
            data.append(data_list)
            data = np.transpose(np.array(data))
            ax.set_ylabel("No. of counts", fontdict=fontdict)

        for i, data_i in enumerate(data):
            ax.bar(x + i * 0.25, data_i, color=colors[i], align="edge", width=0.2)
        ax.legend(
            labels,
            bbox_to_anchor=(1.25, 1),
            loc="upper right",
            prop=fontdict_small,
        )

        ax.set_xticks(
            ticks=x + 0.5,
            labels=list(self.cd.keys()),
            fontsize=fontdict["size"],
        )
        ax.tick_params(axis="y", which="major", labelsize=fontdict["size"])
        plt.show()

        return fig


def main():
    """Plot of data class distribution."""
    np.random.seed(502)
    input_filepath = r"C:\Users\pielsticker\Simulations\20220624_Fe_linear_combination_small_gas_phase\20220624_Fe_linear_combination_small_gas_phase.h5"

    datahandler = DataHandler(intensity_only=False)
    train_test_split = 0.2
    train_val_split = 0.2
    no_of_examples = 200000

    _ = datahandler.load_data_preprocess(
        input_filepath=input_filepath,
        no_of_examples=no_of_examples,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )
    print("Input shape: " + str(datahandler.input_shape))
    print("Labels: " + str(datahandler.labels))
    print("No. of classes: " + str(datahandler.num_classes))

    datahandler.plot_random(no_of_spectra=15, dataset="train", with_prediction=False)

    data_list = [
        datahandler.y,
        datahandler.y_train,
        datahandler.y_val,
        datahandler.y_test,
    ]
    class_distribution = UpdatedClassDistribution(
        task="regression", data_list=data_list
    )

    labels_legend = ["Fe metal", "FeO", "$Fe_{3}O_{4}$", "$Fe_{2}O_{3}$"]
    fig = class_distribution.plot(labels=labels_legend)

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "label_distribution" + ext)
        fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
