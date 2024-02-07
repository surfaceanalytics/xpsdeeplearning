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
Visualization of XPS data set simulation.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

# noqa: E402
from xpsdeeplearning.simulation.base_model.spectra import (
    MeasuredSpectrum,
    SimulatedSpectrum,
)
from xpsdeeplearning.simulation.sim import Simulation

from common import REPO_PATH, SAVE_DIR


def main():
    """Visualization of XPS data set simulation."""
    input_datafolder = os.path.join(
        REPO_PATH, "xpsdeeplearning", "data", "references", "NiCoFe"
    )

    filenames = [
        "Fe2p_Fe_metal.txt",
        "Fe2p_FeO.txt",
        "Fe2p_Fe3O4.txt",
        "Fe2p_Fe2O3.txt",
    ]
    measured_spectra = []
    for filename in filenames:
        filepath = os.path.join(input_datafolder, filename)
        measured_spectra += [MeasuredSpectrum(filepath)]

    fontdict = {"size": 34}
    fontdict_legend = {"size": 22}

    fig = plt.figure(figsize=(36, 18), dpi=300)
    outer = gridspec.GridSpec(1, 3, wspace=0.6, hspace=0.5)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1], wspace=0.2, hspace=0.4
    )

    ax0 = fig.add_subplot(outer[0, 0])
    ax2 = fig.add_subplot(outer[0, 2])
    ax0_0 = fig.add_subplot(inner[0, 0])
    ax0_1 = fig.add_subplot(inner[0, 1])
    ax1_0 = fig.add_subplot(inner[1, 0])
    ax1_1 = fig.add_subplot(inner[1, 1])

    # Reference spectra
    ax0.set_title("(a) Reference spectra", fontdict=fontdict)
    ax0.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax0.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax0.tick_params(axis="x", labelsize=fontdict["size"])
    ax0.tick_params(axis="y", which="both", right=False, left=False)
    ax0.set_yticklabels([])
    colors = iter(["black", "green", "blue", "orange"])
    for spectrum in measured_spectra:
        spectrum.resample(start=695.0, stop=740.0, step=0.1)
        spectrum.normalize()
        ax0.plot(spectrum.x, spectrum.lineshape, c=next(colors), linewidth=2)
        ax0.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    ax0.legend(
        ["Fe$^{0}$", "FeO", "Fe$_{3}$O$_{4}$", "Fe$_{2}$O$_{3}$"],
        prop=fontdict,
        loc=2,
    )

    # Sim values
    shift_x_values = list(np.arange(-3, 4, 1))
    shift_x_values = list(np.delete(shift_x_values, shift_x_values.index(0)))

    sim_values = {
        "shift_x": shift_x_values,
        "noise": [5, 15, 25, 50, 100, 200],
        "FWHM": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        "scatterers": {
            # 0: "He",
            # 1: "H2",
            # 2: "N2",
            3: "O2"
        },
        "pressure": [1, 5, 10],
        "distance": [0.5, 0.8, 0.3],
    }
    colors = [
        "cornflowerblue",
        "orangered",
        "forestgreen",
        "gray",
        "teal",
        "darkviolet",
    ]
    alpha = 0.65

    # Shifting
    ax0_0.set_title("(b) Shift in Binding Energy", fontdict=fontdict)
    ax0_0.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax0_0.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax0_0.tick_params(axis="x", labelsize=fontdict["size"])
    ax0_0.set_yticklabels([])
    ax0_0.tick_params(axis="y", which="both", right=False, left=False)
    legend = []
    for i, shift_x in enumerate(sim_values["shift_x"]):
        spectrum = SimulatedSpectrum(
            start=measured_spectra[0].start,
            stop=measured_spectra[0].stop,
            step=measured_spectra[0].step,
            label=measured_spectra[0].label,
        )
        spectrum.lineshape = measured_spectra[0].lineshape
        spectrum.resample(start=695.0, stop=740.0, step=0.1)
        if i == 0:
            ax0_0.plot(spectrum.x, spectrum.lineshape, c="red", linewidth=2)
            legend.append("original")
        spectrum.shift_horizontal(shift_x)
        ax0_0.plot(
            spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha, linewidth=2
        )
        ax0_0.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
        legend.append(f"x = {shift_x}")
    legend0_0 = ax0_0.legend(
        legend,
        title="Fe$^{0}$, shifted by\nx eV",
        prop=fontdict_legend,
        title_fontsize=fontdict_legend["size"],
        handlelength=1,
        loc=2,
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        shadow=False,
        mode=None,
    )
    legend0_0._legend_box.align = "left"

    # Noise
    ax0_1.set_title("(c) Artificial noise", fontdict=fontdict)
    ax0_1.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax0_1.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax0_1.tick_params(axis="x", labelsize=fontdict["size"])
    ax0_1.set_yticklabels([])
    ax0_1.tick_params(axis="y", which="both", right=False, left=False)
    legend = []
    for i, noise in enumerate(sim_values["noise"]):
        spectrum = SimulatedSpectrum(
            start=measured_spectra[0].start,
            stop=measured_spectra[0].stop,
            step=measured_spectra[0].step,
            label=measured_spectra[0].label,
        )
        spectrum.lineshape = measured_spectra[0].lineshape
        spectrum.resample(start=695.0, stop=740.0, step=0.1)
        spectrum.normalize()
        if i == 0:
            ax0_1.plot(spectrum.x, spectrum.lineshape, c="red", linewidth=2)
            legend.append("original")
        spectrum.add_noise(noise)
        spectrum.normalize()
        ax0_1.plot(
            spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha, linewidth=2
        )
        ax0_1.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
        legend.append(str(noise))
    legend0_1 = ax0_1.legend(
        legend,
        title="Fe$^{0}$, noise changed\n(new S/N ratio)",
        prop=fontdict_legend,
        title_fontsize=fontdict_legend["size"],
        handlelength=1,
        loc=2,
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=False,
        shadow=False,
        mode=None,
    )
    legend0_1._legend_box.align = "left"

    # Resolution
    ax1_0.set_title("(d) Broadening", fontdict=fontdict)
    ax1_0.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax1_0.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax1_0.tick_params(axis="x", labelsize=fontdict["size"])
    ax1_0.set_yticklabels([])
    ax1_1.tick_params(axis="y", which="both", right=False, left=False)
    legend = []
    for i, fwhm in enumerate(sim_values["FWHM"]):
        spectrum = SimulatedSpectrum(
            start=measured_spectra[0].start,
            stop=measured_spectra[0].stop,
            step=measured_spectra[0].step,
            label=measured_spectra[0].label,
        )
        spectrum.lineshape = measured_spectra[0].lineshape
        spectrum.resample(start=695.0, stop=740.0, step=0.1)
        spectrum.normalize()
        if i == 0:
            ax1_0.plot(spectrum.x, spectrum.lineshape, c="red", linewidth=2)
            legend.append("original")

        spectrum.change_resolution(fwhm)
        spectrum.normalize()
        ax1_0.plot(
            spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha, linewidth=2
        )
        ax1_0.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
        legend.append(f"{np.round(fwhm,0)} eV")
    legend1_0 = ax1_0.legend(
        legend,
        title="Fe$^{0}$, broadened\n(FWHM of Gaussian)",
        prop=fontdict_legend,
        title_fontsize=fontdict_legend["size"],
        loc=2,
        handlelength=1,
        ncol=1,
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        frameon=False,
        shadow=False,
        mode=None,
    )
    legend1_0._legend_box.align = "left"

    # Scattering
    ax1_1.set_title("(e) Gas phase scattering", fontdict=fontdict)
    ax1_1.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax1_1.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax1_1.tick_params(axis="x", labelsize=fontdict["size"])
    ax1_1.set_yticklabels([])
    ax1_1.tick_params(axis="y", which="both", right=False, left=False)
    legend = []
    for i in range(len(sim_values["pressure"])):
        spectrum = SimulatedSpectrum(
            start=measured_spectra[0].start,
            stop=measured_spectra[0].stop,
            step=measured_spectra[0].step,
            label=measured_spectra[0].label,
        )
        spectrum.lineshape = measured_spectra[0].lineshape
        spectrum.resample(start=695.0, stop=740.0, step=0.1)
        spectrum.normalize()
        if i == 0:
            ax1_1.plot(spectrum.x, spectrum.lineshape, c="red", linewidth=2)
            ax1_1.set_ylim(None, top=np.max(spectrum.lineshape) * 1.2)
            legend.append("original")
        scatterer = "O2"
        pressure = sim_values["pressure"][i]
        distance = sim_values["distance"][i]
        spectrum.scatter_in_gas(label=scatterer, distance=distance, pressure=pressure)
        spectrum.normalize()
        ax1_1.plot(
            spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha, linewidth=2
        )
        ax1_1.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
        legend.append(f"d = {distance} mm,\nP = {pressure} mbar")

    legend1_1 = ax1_1.legend(
        legend,
        title=f"Fe$^{0}$, scattered in O$_2$",
        prop=fontdict_legend,
        title_fontsize=fontdict_legend["size"],
        loc=2,
        handlelength=1,
        ncol=1,
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        frameon=False,
        shadow=False,
        mode=None,
    )
    legend1_1._legend_box.align = "left"

    #  Linear combination
    colors = iter(["tab:orange", "tab:purple", "grey", "turquoise"])

    sim_params = {
        "0": {
            "linear_params": [0.25, 0.25, 0.25, 0.25],
            "shift_x": 3,
            "noise": 50,
            "FWHM": 2,
            "scatterer": "O2",
            "pressure": 1,
            "distance": 1,
        },
        "1": {
            "linear_params": [0.5, 0.25, 0.25, 0],
            "shift_x": -3,
            "noise": 20,
            "FWHM": 1,
            "scatterer": "O2",
            "pressure": 5,
            "distance": 0.2,
        },
        "2": {
            "linear_params": [0.5, 0, 0, 0.5],
            "shift_x": 3,
            "noise": 10,
            "FWHM": 0,
            "scatterer": "H2",
            "pressure": 5,
            "distance": 0.1,
        },
        "3": {
            "linear_params": [0, 0.75, 0.25, 0],
            "shift_x": 5,
            "noise": 25,
            "FWHM": 0,
            "scatterer": None,
            "pressure": 0,
            "distance": 0,
        },
    }

    ax2.set_title("(f) Simulated spectra", fontdict=fontdict)
    ax2.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax2.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax2.tick_params(axis="x", labelsize=fontdict["size"])
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="both", right=False, left=False)
    legend = []
    for sim_vals in sim_params.values():
        sim = Simulation(measured_spectra)
        sim.combine_linear(sim_vals["linear_params"])
        sim.change_spectrum(
            fwhm=sim_vals["FWHM"],
            shift_x=sim_vals["shift_x"],
            signal_to_noise=sim_vals["noise"],
            scatterer={
                "label": sim_vals["scatterer"],
                "distance": sim_vals["distance"],
                "pressure": sim_vals["pressure"],
            },
        )
        sim.output_spectrum.normalize()
        ax2.plot(
            sim.output_spectrum.x,
            sim.output_spectrum.lineshape,
            c=next(colors),
            linewidth=2,
        )

        # Only use labels in legend if they were used in the simulation.
        labels = ["Fe$^{0}$", "FeO", "Fe$_{3}$O$_{4}$", "Fe$_{2}$O$_{3}$"]
        legend_text = ""
        counter = 0
        for linear_param, label in zip(sim_vals["linear_params"], labels):
            if linear_param > 0:
                counter += 1
                if counter == 3:
                    legend_text += "\n"
                legend_text += f"{int(linear_param*100)} % {label}, "
        legend_text = legend_text.rstrip(", ")
        legend.append(legend_text)

        ax2.set_xlim(
            left=np.max(sim.output_spectrum.x),
            right=np.min(sim.output_spectrum.x),
        )
    legend2 = ax2.legend(
        legend,
        prop={"size": 28},
        loc=0,
        ncol=1,
        bbox_to_anchor=(0, 0, 1, 1),
        fancybox=True,
        frameon=True,
        shadow=False,
        mode=None,
    )
    legend2._legend_box.align = "left"

    outer.tight_layout(fig)
    fig.show()

    for ext in [".png", ".eps"]:
        fig_path = os.path.join(SAVE_DIR, "sim_visualization" + ext)
        fig.savefig(fig_path, bbox_inches="tight")

    # Only linear combination
    fig, ax3 = plt.subplots(figsize=(5, 4), dpi=300)
    ax3.set_xlabel("Binding energy (eV)", fontdict=fontdict)
    ax3.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
    ax3.tick_params(axis="x", labelsize=fontdict["size"])
    ax3.set_yticklabels([])
    ax3.tick_params(axis="y", which="both", right=False, left=False)
    colors = iter(["tab:orange", "tab:purple", "grey", "turquoise"])
    for sim_values in sim_params.values():
        sim = Simulation(measured_spectra)
        sim.combine_linear(sim_values["linear_params"])
        sim.output_spectrum.normalize()
        ax3.plot(
            sim.output_spectrum.x,
            sim.output_spectrum.lineshape,
            c=next(colors),
            linewidth=2,
        )
    ax3.set_xlim(
        left=np.max(sim.output_spectrum.x), right=np.min(sim.output_spectrum.x)
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(__file__).split("deepxps")[0], "deepxps"))
    main()
