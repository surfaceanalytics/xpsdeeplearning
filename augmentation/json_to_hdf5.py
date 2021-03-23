# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:03:47 2020

@author: pielsticker

This script is used to combine data that was simulated and subsequently
stored in many JSON files and store it in one HDF5 file.
""" 

import os
import numpy as np
import json
import h5py
from time import time

from creator import calculate_runtime

#%%
def load_data_preprocess(json_datafolder,
                         label_list,
                         start,
                         end):
    """
    This function loads data from JSON files in the json_datafolder.
    Data from the files that start and end with the respective numbers
    is loaded.

    Parameters
    ----------
    json_datafolder : str
        Folderpath for the JSON files.
    label_list : list
        List of strings with label names.
    start : int
        First file to load.
    end : int
        Last file to load.

    Returns
    -------
    X : arr
        3D numpy array in the format (no_of_spectra, len of 1 spectrum, 1).
    y : arr
        One-hot encoded labels. Label values need to be given/changed
        in the first line of this method.        
    shiftx : arr
        Array of float values of the shiftx values.
    noise : arr
        Array of float values of the noise values.
    FWHM : arr
        Array of float values of the fwhm values.
    scatterer : arr
        Array of float values describing each scatterer. Translation
        has to be available.
    distance : arr
        Array of float values of the distance values.
    pressure : pressure
        Array of float values of the pressure values.

    """   
    filenames = next(os.walk(json_datafolder))[2]
    try:
        filenames.remove('run_params.json')
    except ValueError:
        pass
    X = []
    y = []
    shiftx = []
    noise = []
    FWHM = []
    scatterer = []
    distance = []
    pressure = []
        
    for file in filenames[start:end]:
        filename = os.path.join(json_datafolder, file)
        with open(filename, 'r') as json_file:
            test = json.load(json_file)
        for j, spec_data in enumerate(test):
            X_one = spec_data['y']
            y_one = spec_data['label']
            shiftx_one = spec_data['shift_x']
            noise_one = spec_data['noise']
            FWHM_one = spec_data['FWHM']
            scatterer_name = spec_data['scatterer']
            scatterers = {'He' : 0, 'H2' : 1, 'N2' : 2, 'O2' : 3}
            scatterer_one = scatterers[scatterer_name]
            distance_one = spec_data['distance']
            pressure_one = spec_data['pressure']

            X.append(X_one)
            y.append(y_one)
            shiftx.append(shiftx_one)
            noise.append(noise_one)
            FWHM.append(FWHM_one)
            scatterer.append(scatterer_one)
            distance.append(distance_one)
            pressure.append(pressure_one)
            number = j
        print('Load: ' + str((filenames.index(file)-start)*len(test)+number+1) + '/' + \
                      str(len(filenames[start:end])*len(test)))
                                                  
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = _one_hot_encode(y, label_list)
    
    shiftx = np.reshape(np.array(shiftx),(-1,1))
    noise = np.reshape(np.array(noise),(-1,1))
    FWHM = np.reshape(np.array(FWHM),(-1,1))
    scatterer = np.reshape(np.array(scatterer),(-1,1))
    distance = np.reshape(np.array(distance),(-1,1))
    pressure = np.reshape(np.array(pressure),(-1,1))

    return X, y, shiftx, noise, FWHM, scatterer, distance, pressure
        
def _load_energies(filepath):
    """
    Load the energy scale from one json file.

    Parameters
    ----------
    filepath : str
        Filepath of the json file.

    Returns
    -------
    energies : ndarray
        1d array with binding energies.

    """
    with open(filepath, 'r') as json_file:
        test = json.load(json_file)
    energies = np.array(test[0]['x'])
    
    return energies

def _load_labels(filepath):
    """
    Load the list of label values from one json file.

    Parameters
    ----------
    filepath : str
        Filepath of the json file.

    Returns
    -------
    energies : ndarray
        1d array with binding energies.

    """
    with open(filepath, 'r') as json_file:
        test = json.load(json_file)
    
    labels = list(test[0]['label'].keys())    
    return labels

  
def _one_hot_encode(y,
                    label_list):
    """
    One-hot encode the labels.
    As an example, if the label of a spectrum is Fe metal = 1 and all 
    oxides = 0, then the output will be np.array([1,0,0,0],1).

    Parameters
    ----------
    y : list
        List of label strings.
    label_list : list
        List of strings with label names.

    Returns
    -------
    new_labels : arr
        One-hot encoded labels.

    """    
    new_labels = np.zeros((len(y), len(label_list)))    
    
    for i,d in enumerate(y):
        for species, value in d.items():  
            number = label_list.index(species)
            new_labels[i, number] = value
          
    return new_labels


def to_hdf5(json_datafolder,
            output_file,
            no_of_files_per_load = 50):
    """
    Function to store all data in an input datafolder in an HDF5
    file.    

    Parameters
    ----------
    json_datafolder : str
        Path where the json files are located.
    output_file : str
        Output file name (.h5).
    no_of_files_per_load : int
        Number of files to load before the HDF5 file is updated.
        Typically around 50.


    Returns
    -------
    None.

    """
    filenames = next(os.walk(json_datafolder))[2]
    no_of_files = len(filenames)
    no_of_loads = int(no_of_files/no_of_files_per_load)
    
    with h5py.File(output_file, 'w') as hf:
        # Store energies and labels in separate dataset.
        filepath = os.path.join(json_datafolder,
                                filenames[0])
        energies = _load_energies(filepath)
        hf.create_dataset('energies', data = energies,
                          compression="gzip", chunks=True)
        
        label_list = _load_labels(filepath)   
        labels = np.array(label_list, dtype=object)  
        string_dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('labels', data = labels,
                          dtype=string_dt,
                          compression="gzip", chunks=True)
        
        start = 0
        end = no_of_files_per_load
        X, y, shiftx, noise, FWHM, scatterer, distance, pressure = \
            load_data_preprocess(json_datafolder,
                                 label_list,
                                 start,
                                 end)
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
        hf.create_dataset('scatterer', data = scatterer,
                          compression="gzip", chunks=True,
                          maxshape=(None, scatterer.shape[1]))
        hf.create_dataset('distance', data = distance,
                          compression="gzip", chunks=True,
                          maxshape=(None, distance.shape[1]))
        hf.create_dataset('pressure', data = pressure,
                          compression="gzip", chunks=True,
                          maxshape=(None, pressure.shape[1]))
        print('Saved: ' + str(1) + '/' + str(no_of_loads))
                
        for load in range(1,no_of_loads):
            start = load*no_of_files_per_load
            end = start+no_of_files_per_load
            X_new, y_new, shiftx_new, noise_new, FWHM_new, \
                scatterer_new, distance_new, pressure_new  = \
                    load_data_preprocess(json_datafolder,
                                         label_list,
                                         start,
                                         end)
            
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
            
            
            hf["scatterer"].resize((hf["scatterer"].shape[0] + 
                               scatterer_new.shape[0]), axis = 0)
            hf["scatterer"][-scatterer_new.shape[0]:] = scatterer_new
            hf["distance"].resize((hf["distance"].shape[0] + 
                               distance_new.shape[0]), axis = 0)
            hf["distance"][-distance_new.shape[0]:] = distance_new
            
            hf["pressure"].resize((hf["pressure"].shape[0] + 
                               pressure_new.shape[0]), axis = 0)
            hf["pressure"][-pressure_new.shape[0]:] = pressure_new
                       
    
            print('Saved: ' + str(load+1) + '/' + str(no_of_loads))

#%%               
if __name__ == "__main__":
    json_datafolder = r'C:\Users\pielsticker\Simulations\20210308_Pd_linear_combination_small_gas_phase'
    param_filepath = os.path.join(json_datafolder,
                                  'run_params.json')
    with open(param_filepath, 'r') as param_file:
        params = json.load(param_file)
        
    output_filepath = params['output_datafolder'] + params['h5_filename']

    runtimes = {}
    t0 = time()
    to_hdf5(json_datafolder, output_filepath)
    t1 = time()
    runtimes['h5_save'] = calculate_runtime(t0,t1)
    print('finished saving')
    
    # Test new file.
    t0 = time()    
    with h5py.File(output_filepath, 'r') as hf:
        size = hf['X'].shape
        X_h5 = hf['X'][:4000,:,:]
        y_h5 = hf['y'][:4000,:]
        shiftx_h5 = hf['shiftx'][:4000,:]
        noise_h5 = hf['noise'][:4000,:]
        fwhm_h5 = hf['FWHM'][:4000,:]
        energies_h5 = hf['energies'][:]
        labels_h5 = [str(label) for label in hf['labels'][:]]
    t1 = time()
    runtimes['h5_load'] = calculate_runtime(t0,t1)