# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:50:14 2021

@author: pielsticker
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

os.chdir(
    os.path.join(
        os.path.abspath(__file__).split("deepxps")[0], "deepxps"
    )
)
cw = os.getcwd()

# noqa: E402
from xpsdeeplearning.simulation.base_model.spectra import (
    MeasuredSpectrum,
    SimulatedSpectrum,
)
from xpsdeeplearning.simulation.base_model.figures import Figure
from xpsdeeplearning.simulation.sim import Simulation


# %% Loading
input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\xpsdeeplearning\data\references\NiCoFe"

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

fontdict = {"size": 17}
fontdict_bold = {"size": 17, "weight": "bold"}
fontdict_legend = {"size": 8.5}

fig = plt.figure(figsize=(35, 9), dpi=300)
outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=3)
inner = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[1], wspace=0.4, hspace=0.45
)

ax0 = fig.add_subplot(outer[0, 0])
ax2 = fig.add_subplot(outer[0, 2])
ax0_0 = fig.add_subplot(inner[0, 0])
ax0_1 = fig.add_subplot(inner[0, 1])
ax1_0 = fig.add_subplot(inner[1, 0])
ax1_1 = fig.add_subplot(inner[1, 1])

# %% Reference spectra
# ax0.set_title("a", fontdict=fontdict_bold, loc='left')
ax0.set_title("(a) Reference spectra", fontdict=fontdict)
ax0.set_xlabel("Binding energy (eV)", fontdict=fontdict)
ax0.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
ax0.tick_params(axis="x", labelsize=fontdict["size"])
ax0.set_yticklabels([])
colors = iter(["black", "green", "blue", "orange"])
for spectrum in measured_spectra:
    spectrum.resample(start=695.0, stop=740.0, step=0.1)
    spectrum.normalize()
    ax0.plot(spectrum.x, spectrum.lineshape, c=next(colors))
    ax0.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
ax0.legend(
    ["Fe$^{0}$", "FeO", "Fe$_{3}$O$_{4}$", "Fe$_{2}$O$_{3}$"],
    prop=fontdict,
    loc=2,
)

# %% Sim values
shift_x_values = list(np.arange(-3, 4, 1))
shift_x_values = list(
    np.delete(shift_x_values, shift_x_values.index(0))
)

sim_values = {
    "shift_x": shift_x_values,
    "noise": [5, 15, 25, 50, 100, 200],
    "FWHM": [50, 80, 100, 150, 200, 300],
    "scatterers": {
        "0": "He",
        # "1": "H2",
        # "2": "N2",
        # "3": "O2"
    },
    "pressure": {"0": [1, 20], "1": [1, 7]},
    "distance": [0.1, 1],
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

# %% Shifting
# ax0_0.set_title("b", fontdict=fontdict_bold, loc='left')
ax0_0.set_title("(b) Horizontal shift", fontdict=fontdict)
ax0_0.set_xlabel("Binding energy (eV)", fontdict=fontdict)
ax0_0.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
ax0_0.tick_params(axis="x", labelsize=fontdict["size"])
ax0_0.set_yticklabels([])
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
        ax0_0.plot(spectrum.x, spectrum.lineshape, c="red")
        legend.append("original")
    spectrum.shift_horizontal(shift_x)
    ax0_0.plot(spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha)
    ax0_0.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    legend.append(f"x = {shift_x}")
legend0_0 = ax0_0.legend(
    legend,
    title="Fe$^{0}$, shifted by x eV",
    prop=fontdict_legend,
    title_fontsize=fontdict_legend["size"],
    loc=2,
    ncol=1,
    frameon=False,
    bbox_to_anchor=(0, 0, 1, 1),
    fancybox=True,
    shadow=False,
    mode=None,
)
legend0_0._legend_box.align = "left"

#%% Noise
# ax0_1.set_title("c", fontdict=fontdict_bold, loc='left')
ax0_1.set_title("(c) Artificial noise", fontdict=fontdict)
ax0_1.set_xlabel("Binding energy (eV)", fontdict=fontdict)
ax0_1.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
ax0_1.tick_params(axis="x", labelsize=fontdict["size"])
ax0_1.set_yticklabels([])
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
        ax0_1.plot(spectrum.x, spectrum.lineshape, c="red")
        legend.append("original")
    spectrum.add_noise(noise)
    spectrum.normalize()
    ax0_1.plot(spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha)
    ax0_1.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    legend.append(str(noise))
legend0_1 = ax0_1.legend(
    legend,
    title="Fe$^{0}$, noise changed (new S/N ratio)",
    prop=fontdict_legend,
    title_fontsize=fontdict_legend["size"],
    loc=2,
    ncol=1,
    frameon=False,
    bbox_to_anchor=(0, 0, 1, 1),
    fancybox=False,
    shadow=False,
    mode=None,
)
legend0_1._legend_box.align = "left"

#%% Resolution
# ax1_0.set_title("d", fontdict=fontdict_bold, loc='left')
ax1_0.set_title("(d) Change of resolution", fontdict=fontdict)
ax1_0.set_xlabel("Binding energy (eV)", fontdict=fontdict)
ax1_0.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
ax1_0.tick_params(axis="x", labelsize=fontdict["size"])
ax1_0.set_yticklabels([])
legend = []
for i, FWHM in enumerate(sim_values["FWHM"]):
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
        ax1_0.plot(spectrum.x, spectrum.lineshape, c="red")
        legend.append("original")
    spectrum.change_resolution(FWHM)
    spectrum.normalize()
    ax1_0.plot(spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha)
    ax1_0.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    legend.append(f"{FWHM} eV")
legend1_0 = ax1_0.legend(
    legend,
    title="Fe$^{0}$, resolution changed (new FWHM)",
    prop=fontdict_legend,
    title_fontsize=fontdict_legend["size"],
    loc=2,
    ncol=1,
    bbox_to_anchor=(0, 0, 1, 1),
    fancybox=True,
    frameon=False,
    shadow=False,
    mode=None,
)
legend1_0._legend_box.align = "left"


#%% Scattering
def _select_random_scatterer_key(sim_values):
    return np.random.randint(0, len(sim_values["scatterers"].keys()))


def _select_random_scatterer(sim_values, key):
    return sim_values["scatterers"][str(key)]


def _select_random_scatter_pressure(sim_values, key):
    sim_range = sim_values["pressure"][str(key)]
    return np.random.randint(sim_range[0] * 10, sim_range[1] * 10,) / 10


def _select_random_scatter_distance(sim_values):
    return (
        np.random.randint(
            sim_values["distance"][0] * 100,
            sim_values["distance"][1] * 100,
        )
        / 100
    )


# ax1_1.set_title("e", fontdict=fontdict_bold, loc='left')
ax1_1.set_title("(e) Scattering in gas phase", fontdict=fontdict)
ax1_1.set_xlabel("Binding energy (eV)", fontdict=fontdict)
ax1_1.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
ax1_1.tick_params(axis="x", labelsize=fontdict["size"])
ax1_1.set_yticklabels([])
legend = []
for i in range(0, 4):
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
        ax1_1.plot(spectrum.x, spectrum.lineshape, c="red")
        legend.append("original")
    key = _select_random_scatterer_key(sim_values)
    scatterer = _select_random_scatterer(sim_values, key)
    pressure = _select_random_scatter_pressure(sim_values, key)
    distance = _select_random_scatter_distance(sim_values)
    spectrum.scatter_in_gas(
        label=scatterer, distance=distance, pressure=pressure
    )
    spectrum.normalize()
    ax1_1.plot(spectrum.x, spectrum.lineshape, c=colors[i], alpha=alpha)
    ax1_1.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    legend.append(f"d: {distance} mm, P: {pressure} mbar")
legend1_1 = ax1_1.legend(
    legend,
    title="Fe$^{0}$, scattered in He",
    prop=fontdict_legend,
    title_fontsize=fontdict_legend["size"],
    loc=2,
    ncol=1,
    bbox_to_anchor=(0, 0, 1, 1),
    fancybox=True,
    frameon=False,
    shadow=False,
    mode=None,
)
legend1_1._legend_box.align = "left"

#%% Linear combination
# colors = iter(["m", "tab:purple", "grey", "black"])
colors = iter(["tab:orange", "tab:purple", "grey", "turquoise"])

sim_params = {
    "0": {
        "linear_params": [0.25, 0.25, 0.25, 0.25],
        "shift_x": 3,
        "noise": 50,
        "FWHM": 500,
        "scatterer": "O2",
        "pressure": 1,
        "distance": 1,
    },
    "1": {
        "linear_params": [0.5, 0.25, 0.25, 0],
        "shift_x": -6,
        "noise": 20,
        "FWHM": 1000,
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

# ax1_1.set_title("f", fontdict=fontdict_bold, loc='left')
ax2.set_title("(f) Simulated spectra", fontdict=fontdict)
ax2.set_xlabel("Binding energy (eV)", fontdict=fontdict)
ax2.set_ylabel("Intensity (arb. units)", fontdict=fontdict)
ax2.tick_params(axis="x", labelsize=fontdict["size"])
ax2.set_yticklabels([])
legend = []
for sim_values in sim_params.values():
    sim = Simulation(measured_spectra)
    sim.combine_linear(sim_values["linear_params"])
    sim.change_spectrum(
        fwhm=sim_values["FWHM"],
        shift_x=sim_values["shift_x"],
        signal_to_noise=sim_values["noise"],
        scatterer={
            "label": sim_values["scatterer"],
            "distance": sim_values["distance"],
            "pressure": sim_values["pressure"],
        },
    )
    sim.output_spectrum.normalize()
    ax2.plot(
        sim.output_spectrum.x,
        sim.output_spectrum.lineshape,
        c=next(colors),
    )

    legend.append(
        f"{int(sim_values['linear_params'][0]*100)} % Fe$^{0}$, "
        + f"{int(sim_values['linear_params'][1]*100)} % FeO, "
        + f"{int(sim_values['linear_params'][2]*100)} % Fe$_{3}$O$_{4}$, "
        + f"{int(sim_values['linear_params'][3]*100)} % Fe$_{2}$O$_{3}$"
    )
    ax2.set_xlim(
        left=np.max(sim.output_spectrum.x),
        right=np.min(sim.output_spectrum.x),
    )
legend2 = ax2.legend(
    legend,
    prop={"size": 13},
    loc=0,
    ncol=1,
    bbox_to_anchor=(0, 0, 1, 1),
    fancybox=True,
    frameon=False,
    shadow=False,
    mode=None,
)
legend1_1._legend_box.align = "left"

fig.tight_layout()
fig.show()

# %% Only linear combination
# =============================================================================
# fig, ax3 = plt.subplots(figsize=(5, 4), dpi=300)
# ax3.set_xlabel("Binding energy (eV)",
#               fontdict=fontdict)
# ax3.set_ylabel("Intensity (arb. units)",
#               fontdict=fontdict)
# ax3.tick_params(axis="x", labelsize=fontdict["size"])
# ax3.set_yticklabels([])
# colors = iter(["tab:orange","tab:purple","grey","turquoise"])
# for sim_values in sim_params.values():
#     sim = Simulation(measured_spectra)
#     sim.combine_linear(sim_values["linear_params"])
#     sim.output_spectrum.normalize()
#     ax3.plot(sim.output_spectrum.x, sim.output_spectrum.lineshape, c=next(colors))
# ax3.set_xlim(left=np.max(sim.output_spectrum.x),
#             right=np.min(sim.output_spectrum.x))
# fig.tight_layout()
# plt.show()
# =============================================================================
