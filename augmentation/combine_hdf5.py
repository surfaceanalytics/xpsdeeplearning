# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:03:47 2020

@author: pielsticker


This script can be used to combine two datasets with data of the same
sahpe. A typical use case would be two data sets created by simulations
from different base reference spectra that shall be used as a combined
training/test data set.

"""
import os
import numpy as np
import h5py
from sklearn.utils import shuffle

#%%

def load_data(filenames):
    input_datafolder = r'C:\Users\pielsticker\Simulations'
    for filename in filenames[:1]:
        input_filepath = os.path.join(input_datafolder,filename)
        with h5py.File(input_filepath, 'r') as hf:
            X = hf['X'][:,:,:]
            y = hf['y'][:,:]
            shiftx = hf['shiftx'][:]
            noise = hf['noise'][:]
            fwhm = hf['FWHM'][:]
            scatterer = hf['scatterer'][:]
            distance = hf['distance'][:]
            pressure = hf['pressure'][:]
            print('File 0 loaded')

    for filename in filenames[1:]:
        input_filepath = os.path.join(input_datafolder,filename)
        with h5py.File(input_filepath, 'r') as hf:
            X_new = hf['X'][:,:,:]
            y_new = hf['y'][:,:]
            shiftx_new = hf['shiftx'][:]
            noise_new = hf['noise'][:]
            fhwm_new = hf['FWHM'][:]
            scatterer_new = hf['scatterer'][:]
            distance_new = hf['distance'][:]
            pressure_new = hf['pressure'][:]
            
            X = np.concatenate((X, X_new), axis = 0)
            y = np.concatenate((y, y_new), axis = 0)
            shiftx = np.concatenate((shiftx, shiftx_new), axis = 0)
            noise = np.concatenate((noise, noise_new), axis = 0)
            fwhm = np.concatenate((fwhm, fhwm_new), axis = 0)
            scatterer = np.concatenate((scatterer, scatterer_new), axis = 0)
            distance = np.concatenate((distance, distance_new), axis = 0)
            pressure = np.concatenate((pressure, pressure_new), axis = 0)
            print('File {0} loaded'.format(filenames.index(filename)))
         
    return X, y, shiftx, noise, fwhm, scatterer, distance, pressure


filenames = ['20200708_iron_variable_linear_combination_500000.h5',
             '20200714_iron_Mark_variable_linear_combination.h5']

X, y, shiftx, noise, fwhm, \
    scatterer, distance, pressure = load_data(filenames)

# Shuffle all numpy arrays together.
X_shuff, y_shuff, shiftx_shuff, noise_shuff, fwhm_shuff,\
    scatterer_shuff, distance_shuff, pressure_shuff = \
        shuffle(X, y, shiftx, noise, fwhm, scatterer, distance, pressure)


#%%    
output_file = r'C:\Users\pielsticker\Simulations\20200720_iron_variable_linear_combination_combined_data.h5'   

with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('X', data = X_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None,X.shape[1],X.shape[2]))
    print('X written')        
    hf.create_dataset('y', data = y_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, y.shape[1]))
    print('y written')
    hf.create_dataset('shiftx', data = shiftx_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, shiftx.shape[1]))
    print('shift written')
    hf.create_dataset('noise', data = noise_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, noise.shape[1]))
    print('noise written')
    hf.create_dataset('FWHM', data = fwhm_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, fwhm.shape[1]))
    print('fwhm written')
    hf.create_dataset('scatterer', data = scatterer_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, scatterer.shape[1]))
    print('scatterer written')
    hf.create_dataset('distance', data = distance_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, distance.shape[1]))
    print('distance written')
    hf.create_dataset('pressure', data = pressure_shuff,
                      compression="gzip", chunks=True,
                      maxshape=(None, pressure.shape[1]))
    print('pressure written')
    
# Test new file.    
with h5py.File(output_file, 'r') as hf:
    size = hf['X'].shape
    X_h5_one = hf['X'][:100,:,:]
    y_h5_one = hf['y'][:100,:]
    shiftx_h5_one = hf['shiftx'][:100]
    noise_h5_one = hf['noise'][:100]
    fwhm_h5_one = hf['FWHM'][:100]        