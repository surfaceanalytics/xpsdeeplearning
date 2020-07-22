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

from creator import calculate_runtime



def load_data_preprocess(input_datafolder,start,end):
    label_values = ['Fe metal','FeO','Fe3O4','Fe2O3']
    
    filenames = next(os.walk(input_datafolder))[2]

    X = []
    y = []
    shiftx = []
    noise = []
    FWHM = []
        
    for file in filenames[start:end]:
        filename = input_datafolder + file
        with open(filename, 'r') as json_file:
            test = json.load(json_file)
        for j in range(0,len(test)):
            X_one = test[j]['y']
            y_one = test[j]['label']
            shiftx_one = test[j]['shift_x']
            noise_one = test[j]['noise']
            FWHM_one = test[j]['FWHM']

            X.append(X_one)
            y.append(y_one)
            shiftx.append(shiftx_one)
            noise.append(noise_one)
            FWHM.append(FWHM_one)
        print('Load: ' + str((filenames.index(file)-start)*len(test)+j+1) + '/' + \
                      str(len(filenames[start:end])*len(test)))
                                                  
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = _one_hot_encode(y, label_values)
    
    shiftx = np.reshape(np.array(shiftx),(-1,1))
    noise = np.reshape(np.array(noise),(-1,1))
    FWHM = np.reshape(np.array(FWHM),(-1,1))

    return X, y, shiftx, noise, FWHM
        
    
def _one_hot_encode(y, label_values):
    new_labels = np.zeros((len(y), len(label_values)))    
    
    for i,d in enumerate(y):
        for species, value in d.items():  
            number = label_values.index(species)
            new_labels[i, number] = value
          
    return new_labels


def to_hdf5(output_file, simulation_name, no_of_files_per_load):
    input_datafolder = r'C:\Users\pielsticker\Simulations\\' + \
        simulation_name + '\\'
    filenames = next(os.walk(input_datafolder))[2]
    no_of_files = len(filenames)    
    no_of_loads = int(no_of_files/no_of_files_per_load) 
    
    with h5py.File(output_file, 'w') as hf:
        start = 0
        end = no_of_files_per_load
        X, y, shiftx, noise, FWHM  = \
            load_data_preprocess(input_datafolder, start, end)
        hf.create_dataset('X', data = X,
                          compression="gzip", chunks=True,
                          maxshape=(None,X.shape[1],X.shape[2]))        
        hf.create_dataset('y', data = y,
                          compression="gzip", chunks=True,
                          maxshape=(None, y.shape[1]))
        hf.create_dataset('shiftx', data = shiftx,
                          compression="gzip", chunks=True,
                          maxshape=(None, shiftx.shape[1]))
        hf.create_dataset('noise', data = noise,
                          compression="gzip", chunks=True,
                          maxshape=(None, noise.shape[1]))
        hf.create_dataset('FWHM', data = FWHM,
                          compression="gzip", chunks=True,
                          maxshape=(None, FWHM.shape[1]))
        print('Saved: ' + str(1) + '/' + str(no_of_loads))
        
        for load in range(1,no_of_loads):
            start = load*no_of_files_per_load
            end = start+no_of_files_per_load
            X_new, y_new, shiftx_new, noise_new, FWHM_new  = \
                load_data_preprocess(input_datafolder, start, end)
            
            hf["X"].resize((hf["X"].shape[0] + X_new.shape[0]), axis = 0)
            hf["X"][-X_new.shape[0]:] = X_new
            hf["y"].resize((hf["y"].shape[0] + y_new.shape[0]), axis = 0)
            hf["y"][-y_new.shape[0]:] = y_new
            
            hf["shiftx"].resize((hf["shiftx"].shape[0] +
                                 shiftx_new.shape[0]), axis = 0)
            hf["shiftx"][-X_new.shape[0]:] = shiftx_new
            
            hf["noise"].resize((hf["noise"].shape[0] +
                                noise_new.shape[0]), axis = 0)
            hf["noise"][-X_new.shape[0]:] = noise_new
            hf["FWHM"].resize((hf["FWHM"].shape[0] + 
                               FWHM_new.shape[0]), axis = 0)
            hf["FWHM"][-X_new.shape[0]:] = FWHM_new
    
            print('Saved: ' + str(load+1) + '/' + str(no_of_loads))

#%%               
if __name__ == "__main__":
    output_datafolder = r'C:\Users\pielsticker\Simulations\\'
    output_file = output_datafolder + '20200714_iron_Mark_variable_linear_combination.h5'
    simulation_name = '20200714_iron_Mark_variable_linear_combination'
    no_of_files_per_load = 50

    runtimes = {}
    t0 = time()
    to_hdf5(output_file, simulation_name, no_of_files_per_load)
    t1 = time()
    runtimes['h5_save_iron_single'] = calculate_runtime(t0,t1)
    print('finished saving')
    
    t0 = time()    
    with h5py.File(output_file, 'r') as hf:
        size = hf['X'].shape
        X_h5 = hf['X'][:4000,:,:]
        y_h5 = hf['y'][:4000,:]
        shiftx_h5 = hf['shiftx'][:4000,:]
        noise_h5 = hf['noise'][:4000,:]
        fwhm_h5 = hf['FWHM'][:4000,:]
        t1 = time()
        runtimes['h5_load_iron_single'] = calculate_runtime(t0,t1)
        