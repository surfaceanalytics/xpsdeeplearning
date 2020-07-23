# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:03:47 2020

@author: pielsticker
"""
import os
import numpy as np
import json
import h5py
from time import time, sleep
from sklearn.utils import shuffle



def load_data(filenames):
    input_datafolder = r'C:\Users\pielsticker\Simulations'
    no_of_examples = 0
    for filename in filenames[:1]:
        input_filepath = os.path.join(input_datafolder,filename)
        with h5py.File(input_filepath, 'r') as hf:
# =============================================================================
#             X = hf['X'][:,:,:]
#             y = hf['X'][:,:]
#             shiftx = hf['shiftx'][:]
#             noise = hf['noise'][:]
#             fwhm = hf['FWHM'][:]
#             y_one = y[0,:]
# =============================================================================
            X = hf['X'][:,:,:]
            y = hf['y'][:,:]
            shiftx = hf['shiftx'][:]
            noise = hf['noise'][:]
            fwhm = hf['FWHM'][:]
            y_one = y[0:1,:]
            print('File 0 loaded')

    for filename in filenames[1:]:
        input_filepath = os.path.join(input_datafolder,filename)
        with h5py.File(input_filepath, 'r') as hf:
# =============================================================================
#             X = np.concatenate(X, hf['X'][:,:,:])
#             y = np.concatenate(y, hf['y'][:,:])
#             shiftx = np.concatenate(shiftx, hf['shiftx'][:])
#             noise = np.concatenate(noise, hf['noise'][:])
#             fwhm = np.concatenate(fwhm, hf['FWHM'][:])
# =============================================================================
            X_new = hf['X'][:,:,:]
            y_new = hf['y'][:,:]
            shiftx_new = hf['shiftx'][:]
            noise_new = hf['noise'][:]
            fhwm_new = hf['FWHM'][:]
            
            X = np.concatenate((X, X_new), axis = 0)
            y = np.concatenate((y, y_new), axis = 0)
            shiftx = np.concatenate((shiftx, shiftx_new), axis = 0)
            noise = np.concatenate((noise, noise_new), axis = 0)
            fwhm = np.concatenate((fwhm, fhwm_new), axis = 0)
            y_two = hf['y'][0:1,:]
            print('File {0} loaded'.format(filenames.index(filename)))
         
    return X, y, shiftx, noise, fwhm, y_one, y_two


filenames = ['20200708_iron_variable_linear_combination_500000.h5',
             '20200714_iron_Mark_variable_linear_combination.h5']

X, y, shiftx, noise, fwhm, y_one, y_two = load_data(filenames)

X_shuff, y_shuff, shiftx_shuff, noise_shuff, fwhm_shuff = \
    shuffle(X, y, shiftx, noise, fwhm)

print(np.allclose(y_two,y[500000,:]))

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
    
with h5py.File(output_file, 'r') as hf:
    size = hf['X'].shape
    X_h5_one = hf['X'][:100,:,:]
    y_h5_one = hf['y'][:100,:]
    shiftx_h5_one = hf['shiftx'][:100]
    noise_h5_one = hf['noise'][:100]
    fwhm_h5_one = hf['FWHM'][:100]

print(size)    
        
