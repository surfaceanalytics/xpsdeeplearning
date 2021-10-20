# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:50:14 2021

@author: pielsticker
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from base_model.spectra import MeasuredSpectrum, SimulatedSpectrum
from base_model.figures import Figure
import matplotlib.colors as mcolors
from sim import Simulation

#%% Loading
input_datafolder = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\xpsdeeplearning\data\references\NiCoFe"

filenames = [
    "Fe2p_Fe_metal.txt",
    "Fe2p_FeO.txt",
    "Fe2p_Fe3O4.txt",
    "Fe2p_Fe2O3.txt"
]
measured_spectra = []
for filename in filenames:
    filepath = os.path.join(input_datafolder, filename)
    measured_spectra += [MeasuredSpectrum(filepath)]
    
fontdict = {"size": 20}

#%% Resampling and plotting of measured spectra
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
ax.tick_params(axis="x", labelsize=fontdict["size"])
ax.set_yticklabels([])
colors =  iter(["black", "green","blue","orange"])
for spectrum in measured_spectra:
    spectrum.resample(start=695.0, stop=740.0, step=0.1)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c=next(colors))
    ax.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    #ax.legend(["original"] + [f"shifted by {s} eV" for s in sim_values["shift_x"]])
fig.tight_layout()
plt.show()

#%% Sim values
sim_values = {
    "shift_x":list(np.arange(-3,3,1)),
    "noise":[5,10,15,25,35,50,100,200,500],
    "FWHM":[80,100,200,300],
    "scatterers":{
        "0":"He",
        #"1":"H2",
        #"2":"N2",
        "1":"O2"
        },
    "pressure":{
        "0":[1,20],
        "1":[1,7]
        },
    "distance":[0.1,1]
    }
color = ["cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue"]
alpha = [0.8, 0.8, 0.8, 0.8]

#%% Shifting
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
ax.tick_params(axis="x", labelsize=fontdict["size"])
ax.set_yticklabels([])
colors = iter(mcolors.TABLEAU_COLORS)
for shift_x in sim_values["shift_x"]:
    spectrum =  SimulatedSpectrum(
        start=measured_spectra[0].start,
        stop=measured_spectra[0].stop, 
        step=measured_spectra[0].step,
        label=measured_spectra[0].label)
    spectrum.lineshape = measured_spectra[0].lineshape
    spectrum.resample(start=695.0, stop=740.0, step=0.1)
    ax.plot(spectrum.x, spectrum.lineshape, c="red")
    spectrum.shift_horizontal(shift_x)
    ax.plot(spectrum.x, spectrum.lineshape, c=color[0], alpha=alpha[0])
    ax.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
fig.tight_layout()
plt.show()
#%% Noise
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
ax.tick_params(axis="x", labelsize=fontdict["size"])
ax.set_yticklabels([])
colors = iter(mcolors.TABLEAU_COLORS)
for noise in sim_values["noise"]:
    spectrum =  SimulatedSpectrum(
        start=measured_spectra[0].start,
        stop=measured_spectra[0].stop, 
        step=measured_spectra[0].step,
        label=measured_spectra[0].label)
    spectrum.lineshape = measured_spectra[0].lineshape
    spectrum.resample(start=695.0, stop=740.0, step=0.1)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c="red")
    spectrum.add_noise(noise)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c=color[1], alpha=alpha[1])
    ax.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
fig.tight_layout()
plt.show()
#%% Resolution
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
ax.tick_params(axis="x", labelsize=fontdict["size"])
ax.set_yticklabels([])
colors = iter(mcolors.TABLEAU_COLORS)
for FWHM in sim_values["FWHM"]:
    spectrum =  SimulatedSpectrum(
        start=measured_spectra[0].start,
        stop=measured_spectra[0].stop, 
        step=measured_spectra[0].step,
        label=measured_spectra[0].label)
    spectrum.lineshape = measured_spectra[0].lineshape
    spectrum.resample(start=695.0, stop=740.0, step=0.1)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c="red")
    spectrum.change_resolution(FWHM)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c=color[2], alpha=alpha[2])
    ax.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
fig.tight_layout()
plt.show()

#%% Scattering
def _select_random_scatterer_key(sim_values):
    return np.random.randint(0, len(sim_values["scatterers"].keys()))
    
def _select_random_scatterer(sim_values, key):
    return sim_values["scatterers"][str(key)]


def _select_random_scatter_pressure(sim_values, key):
    sim_range = sim_values["pressure"][str(key)]
    return (
        np.random.randint(
            sim_range[0] * 10,
            sim_range[1] * 10,
            )
        / 10
        )

def _select_random_scatter_distance(sim_values):
    return (
        np.random.randint(
            sim_values["distance"][0] * 100,
            sim_values["distance"][1] * 100,
            )
        / 100
        )

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
ax.tick_params(axis="x", labelsize=fontdict["size"])
ax.set_yticklabels([])
colors =  iter(mcolors.TABLEAU_COLORS)
for i in range(0,8):
    spectrum =  SimulatedSpectrum(
        start=measured_spectra[0].start,
        stop=measured_spectra[0].stop, 
        step=measured_spectra[0].step,
        label=measured_spectra[0].label)
    spectrum.lineshape = measured_spectra[0].lineshape
    spectrum.resample(start=695.0, stop=740.0, step=0.1)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c="red")
    key = _select_random_scatterer_key(sim_values)
    scatterer = _select_random_scatterer(sim_values, key)
    pressure = _select_random_scatter_pressure(sim_values, key)
    distance = _select_random_scatter_distance(sim_values)
    spectrum.scatter_in_gas(
        label=scatterer, 
        distance=distance,
        pressure=pressure)
    spectrum.normalize()
    ax.plot(spectrum.x, spectrum.lineshape, c=color[3], alpha=alpha[3])
    ax.set_xlim(left=np.max(spectrum.x), right=np.min(spectrum.x))
    fig.tight_layout()
plt.show()

#%% Linear combination
colors = iter(["m", "tab:purple", "grey", "black"])

sim_params = {
    "0":  {
       "linear_params" : [0.25, 0.25, 0.25, 0.25],
       "shift_x":3,
       "noise":50,
       "FWHM":500,
       "scatterer": "O2",
       "pressure" : 1,
       "distance": 1
       },
    "1":  {
       "linear_params" : [0.5, 0.25, 0.25, 0],
       "shift_x":-6,
       "noise":20,
       "FWHM":1000,
       "scatterer": "O2",
       "pressure" : 5,
       "distance": 0.2
       },
    "2":  {
       "linear_params" : [0.5, 0, 0, 0.5],
       "shift_x":3,
       "noise":10,
       "FWHM":0,
       "scatterer": "H2",
       "pressure" : 5,
       "distance": 0.1
       },
    "3":  {
       "linear_params" : [0, 0.75, 0.25, 0],
       "shift_x":5,
       "noise":25,
       "FWHM":0,
       "scatterer": None,
       "pressure" : 0,
       "distance": 0
       },
    }
   
for sim_values in sim_params.values():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
    ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
    ax.tick_params(axis="x", labelsize=fontdict["size"])
    ax.set_yticklabels([])
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
            }
        )
    sim.output_spectrum.normalize()
    ax.plot(sim.output_spectrum.x, sim.output_spectrum.lineshape, c=next(colors))
    ax.set_xlim(left=np.max(sim.output_spectrum.x),
                right=np.min(sim.output_spectrum.x))
    fig.tight_layout()
    plt.show()
    
#%%
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
ax.set_xlabel("Binding energy (eV)", 
              fontdict=fontdict)
ax.set_ylabel("Intensity (arb. units)", 
              fontdict=fontdict)
ax.tick_params(axis="x", labelsize=fontdict["size"])
ax.set_yticklabels([])
colors = iter(["tab:orange","tab:purple","grey","turquoise"])
for sim_values in sim_params.values():
    sim = Simulation(measured_spectra)
    sim.combine_linear(sim_values["linear_params"])
    sim.output_spectrum.normalize()
    ax.plot(sim.output_spectrum.x, sim.output_spectrum.lineshape, c=next(colors))
ax.set_xlim(left=np.max(sim.output_spectrum.x),
            right=np.min(sim.output_spectrum.x))
fig.tight_layout()
plt.show()