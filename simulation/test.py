# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:56:33 2021

@author: pielsticker
"""
import os
import numpy as np
from base_model.spectra import MeasuredSpectrum
from base_model.figures import Figure

#%%
datapath = r"C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps\xpsdeeplearning\data\references\NiCoFe"
filenames = list(next(os.walk(datapath))[2])
measured_spectra = []
info = {}
#%%
# =============================================================================
# labels = ["Ni2p_Ni_metal",
#           "Co2p_Ni_metal",
#           "Fe2p_Ni_metal",
#           "Ni2p_Co3O4",
#           "Co2p_Co3O4",
#           "Fe2p_Co3O4",]
# =============================================================================

labels = [
    filename
    for filename in filenames
    if filename.split("_")[0] == "Ni2pCo2pFe2p"
]

for label in labels:
    filepath = os.path.join(datapath, label)
    measured_spectrum = MeasuredSpectrum(filepath)
    measured_spectra.append(measured_spectrum)
    info[label] = {
        "x": measured_spectrum.x.shape,
        "y": measured_spectrum.lineshape.shape,
        "label": measured_spectrum.label,
    }
    fig = Figure(
        measured_spectrum.x,
        measured_spectrum.lineshape,
        title=measured_spectrum.label,
    )

#%%
def write(filename, x, lineshape, label):
    """
    Write to a new txt file.

    Parameters
    ----------
    filename : str
        Filename of the new file.
    x : ndarray
        Binding energies.
    lineshape : ndarray
        Intensities.
    label : str
        Label of the spectrum.

    Returns
    -------
    None.

    """
    datapath = r"C:\Users\pielsticker\Downloads\test"
    filepath_new = os.path.join(datapath, filename)
    with open(filepath_new, "w") as file:
        lines = [label + "\n"]
        for i in range(len(x)):
            lines.append(
                str("{:e}".format(x[i]))
                + " "
                + str("{:e}".format(lineshape[i]))
                + "\n"
            )
        file.writelines(lines)


test_label = "Fe_metal.txt"
labels = ["Ni2p_" + test_label, "Co2p_" + test_label, "Fe2p_" + test_label]

for label in labels:
    filepath = os.path.join(datapath, label)
    measured_spectrum = MeasuredSpectrum(filepath)
    measured_spectra.append(measured_spectrum)
    print(label)
    if label.split("_")[0] == "Ni2p":
        x_ni = measured_spectrum.x
        lineshape_ni = measured_spectrum.lineshape

    elif label.split("_")[0] == "Co2p":
        x_co = measured_spectrum.x[1:]
        lineshape_co = measured_spectrum.lineshape[1:]

    elif label.split("_")[0] == "Fe2p":
        x_fe = measured_spectrum.x[1:701]
        lineshape_fe = measured_spectrum.lineshape[1:701]

new_x = np.concatenate([x_ni, x_co, x_fe])
new_lineshape = np.concatenate([lineshape_ni, lineshape_co, lineshape_fe])
new_filename = "Ni2pCo2pFe2p" + "_" + test_label
new_label = "Ni2pCo2pFe2p" + " " + test_label.split(".")[0]
fig = Figure(new_x, new_lineshape, new_label)

write(new_filename, new_x, new_lineshape, new_label)

new_filepath = os.path.join(datapath, new_filename)
test_spectrum = MeasuredSpectrum(new_filepath)
fig = Figure(
    test_spectrum.x, test_spectrum.lineshape, title=test_spectrum.label
)
