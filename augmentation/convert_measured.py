# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:06:37 2020

@author: pielsticker


This script can be used to load reference and fitted XPS spectra and 
resize them according to new start, stop, and step values.

To do this, the data are stored in ReferenceSpectrum or FittedSpectrum 
instances, respectively. Then all points outside the interval 
(start, stop) are removed and, if needed, the lineshape is extended to
reach the start and stop values Finally, the x and lineshape values are
up-/downsampled according to the new step value.

For fitted spectra, it is possible to load the data from multiple .xy
files and store them in a hdf5 file as a test data set for the neural
network studies.
""" 

import numpy as np
import os
import h5py
import pandas as pd

from base_model.spectra import ReferenceSpectrum, FittedSpectrum
from base_model.figures import Figure
  
#%% For one reference spectrum.   
input_datafolder = r'C:\Users\pielsticker\Desktop\Mixed iron spectra'
filename = 'test.txt'
energies = []

filepath = os.path.join(input_datafolder, filename)
ref_spectrum = ReferenceSpectrum(filepath)
Figure(ref_spectrum.x, ref_spectrum.lineshape, title = 'old')
energies.append(ref_spectrum.x[np.argmax(ref_spectrum.lineshape)])
ref_spectrum.resize(start = 694, stop = 750, step = 0.05)
energies.append(ref_spectrum.x[np.argmax(ref_spectrum.lineshape)])
fig = Figure(ref_spectrum.x, ref_spectrum.lineshape, title = 'new')
#ref_spectrum.write(datafolder)
a = ref_spectrum.lineshape
x = ref_spectrum.x
    
#%% For one fitted XPS spectrum
# =============================================================================
# input_datafolder = r'C:\Users\pielsticker\Desktop\Mixed iron spectra\exported'
# filename = 'measured0001.txt'
# energies = []
# 
# filepath = os.path.join(input_datafolder, filename)
# fit_spectrum = FittedSpectrum(filepath)
# Figure(fit_spectrum.x, fit_spectrum.lineshape, title = 'old')
# energies.append(fit_spectrum.x[np.argmax(fit_spectrum.lineshape)])
# fit_spectrum.resize(start = 694, stop = 750, step = 0.05)
# fit_spectrum.normalize()
# Figure(fit_spectrum.x, fit_spectrum.lineshape, title = 'new')
# energies.append(fit_spectrum.x[np.argmax(fit_spectrum.lineshape)])
# a = fit_spectrum.lineshape
# x = fit_spectrum.x
# =============================================================================

    
#%% For multiple fitted XPS spectra
def _get_labels(): 
    filepath = r'C:\Users\pielsticker\Desktop\Mixed iron spectra\peak fits.xlsx'     
    df = pd.read_excel(filepath)
    y = np.transpose(np.array([df['Fe metal'],
                               df['FeO'],
                               df['Fe3O4'],
                               df['Fe2O3']]))
    names = np.reshape(np.array(list(df['name'])),(-1,1))
                        
    return y, names

def convert_all_spectra(input_datafolder, plot_all = True):
    import warnings
    warnings.filterwarnings("ignore")
    filenames = next(os.walk(input_datafolder))[2]
    
    X = np.zeros((len(filenames),1121,1))
    
    y, names = _get_labels()
    spectra = []
    energies = np.zeros((len(filenames),2))
    
    for name in filenames:
        filepath = os.path.join(input_datafolder, name)
        spectrum = FittedSpectrum(filepath)
        index = filenames.index(name)
        energies[index,0] = spectrum.x[np.argmax(spectrum.lineshape)]
        energies[index,1] = spectrum.x[np.argmax(spectrum.lineshape)]
        spectrum.resize(start = 694, stop = 750, step = 0.05)
        spectrum.normalize()
        spectra.append(spectrum)
        X[index] = np.reshape(spectrum.lineshape, (-1,1))
        

        if plot_all:
            text = 'Spectrum no. ' + str(spectrum.number) + '\n' +\
                str(spectrum.label)
            Figure(spectrum.x, spectrum.lineshape, title = text)
        
    return X, y, names, energies

# Load the data into numpy arrays and save to hdf5 file.
input_datafolder = r'C:\Users\pielsticker\Desktop\Mixed iron spectra\exported'
X, y, names, energies = convert_all_spectra(input_datafolder, plot_all = True)
output_file= r'C:\Users\pielsticker\Simulations\measured.h5'
  
with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('X', data = X,
                      compression="gzip", chunks=True,
                      maxshape=(None,X.shape[1],X.shape[2]))        
    hf.create_dataset('y', data = y,
                      compression="gzip", chunks=True,
                      maxshape=(None, y.shape[1]))
    hf.create_dataset('names', data=np.array(names, dtype='S'),
                      compression="gzip", chunks=True,
                      maxshape=(None, names.shape[1]))

# Test the new file
with h5py.File(output_file, 'r') as hf:
    size = hf['X'].shape
    X_h5 = hf['X'][:,:,:]
    y_h5 = hf['y'][:,:]
    names_h5 = hf['names'][:]