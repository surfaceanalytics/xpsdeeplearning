# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020

@author: pielsticker

This script is go trough all steps of simulating a training data set
from a number of reference spectra.
1) The spectra are simulated and stored in JSON files.
   The no_of_simulations parameter is used to control the number of
   spectra stored in each JSON file and the no_of_files parameter 
   determines how many JSON files are created.
   Total no. of spectra = no_of_simulations*no_of_files 
   
   In this case, two distinct data sets are created using different
   references.
2) Write to individual HDF5 files.
3) Combine the datasets into one big, shuffled dataset.

The time for each step is stored in the runtimes dictionary.    
""" 

from time import time
import os
import datetime
import h5py
from creator import Creator, calculate_runtime

runtimes = {}
t0_full = time()
#%% Simulation of data using reference spectra
no_of_simulations = 500
no_of_files = 500

t0 = time()
input_filenames =  ['Fe_metal_Mark_shifted','FeO_Mark_shifted',
                    'Fe3O4_Mark_shifted','Fe2O3_Mark_shifted']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'iron_Mark_variable_linear_combination_gas_phase'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
os.makedirs(filepath)
filename_basic = os.path.join(*[filepath,time_and_run_name])

# Create multiple sets of similar spectra with the same settings
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_filenames, single = False)
    creator.run(broaden = False, x_shift = True, noise = True, scatter = True)
    creator.plot_random(1)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full')
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

t1 = time()
runtimes['JSON Mark'] = calculate_runtime(t0,t1)
#%% Simulation of data using reference spectra
no_of_simulations = 500
no_of_files = 500

t0 = time()
input_filenames =  ['Fe_metal','FeO','Fe3O4','Fe2O3']
#timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'iron_variable_linear_combination_gas_phase'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
os.makedirs(filepath)
filename_basic = os.path.join(*[filepath,time_and_run_name])

# Create multiple sets of similar spectra with the same settings
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_filenames, single = False)
    creator.run(broaden = False, x_shift = True, noise = True, scatter = True)
    creator.plot_random(1)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full')
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

t1 = time()
runtimes['JSON Lukas'] = calculate_runtime(t0,t1)
#%% Write to HDF5 files
from json_to_hdf5 import to_hdf5

t0 = time()
output_datafolder = r'C:\Users\pielsticker\Simulations\\'
output_file = output_datafolder + '20201612_iron_Mark_variable_linear_combination_gas_phase.h5'
simulation_name = '20201612_iron_Mark_variable_linear_combination_gas_phase'
no_of_files_per_load = 50

to_hdf5(output_file, simulation_name, no_of_files_per_load)
print('finished saving')

with h5py.File(output_file, 'r') as hf:
    size_mark = hf['X'].shape
    X_h5_mark = hf['X'][:4000,:,:]
    y_h5_mark = hf['y'][:4000,:]
    shiftx_h5_mark = hf['shiftx'][:4000,:]
    noise_h5_mark = hf['noise'][:4000,:]
    fwhm_h5_mark = hf['FWHM'][:4000,:]

t1 = time()
runtimes['HDF5 Mark'] = calculate_runtime(t0,t1)

t0 = time()
output_datafolder = r'C:\Users\pielsticker\Simulations\\'
output_file = output_datafolder + '20201612_iron_variable_linear_combination_gas_phase.h5'
simulation_name = '20201612_iron_variable_linear_combination_gas_phase'
no_of_files_per_load = 50

to_hdf5(output_file, simulation_name, no_of_files_per_load)
print('finished saving')

with h5py.File(output_file, 'r') as hf:
    size_lukas = hf['X'].shape
    X_h5_lukas = hf['X'][:4000,:,:]
    y_h5_lukas = hf['y'][:4000,:]
    shiftx_h5_lukas = hf['shiftx'][:4000,:]
    noise_h5_lukas = hf['noise'][:4000,:]
    fwhm_h5_lukas = hf['FWHM'][:4000,:]

t1 = time()
runtimes['HDF5 Lukas'] = calculate_runtime(t0,t1)    
#%%  Combine the datasets    
import numpy as np
from sklearn.utils import shuffle

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
            fwhm_new = hf['FWHM'][:]
            scatterer_new = hf['scatterer'][:]
            distance_new = hf['distance'][:]
            pressure_new = hf['pressure'][:]
                       
            X = np.concatenate((X, X_new), axis = 0)
            y = np.concatenate((y, y_new), axis = 0)
            shiftx = np.concatenate((shiftx, shiftx_new), axis = 0)
            noise = np.concatenate((noise, noise_new), axis = 0)
            fwhm = np.concatenate((fwhm, fwhm_new), axis = 0)
            scatterer = np.concatenate((scatterer, scatterer_new), axis = 0)
            distance = np.concatenate((distance, distance_new), axis = 0)
            pressure = np.concatenate((pressure, pressure_new), axis = 0)
            
            print('File {0} loaded'.format(filenames.index(filename)+1))
         
    return X, y, shiftx, noise, fwhm, scatterer, distance, pressure


filenames = ['20201612_iron_Mark_variable_linear_combination_gas_phase.h5',
             '20201612_iron_variable_linear_combination_gas_phase.h5']

t0 = time()
X, y, shiftx, noise, fwhm, scatterer, distance, pressure = load_data(filenames)
X_shuff, y_shuff, shiftx_shuff, noise_shuff, fwhm_shuff,\
    scatterer_shuff, distance_shuff, pressure_shuff = \
        shuffle(X, y, shiftx, noise, fwhm, scatterer, distance, pressure)

output_file = r'C:\Users\pielsticker\Simulations\20201612_iron_variable_linear_combination_gas_phase_combined_data.h5'   

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
    
with h5py.File(output_file, 'r') as hf:
    size_out = hf['X'].shape
    X_h5_out = hf['X'][:100,:,:]
    y_h5_out= hf['y'][:100,:]
    shiftx_h5_out = hf['shiftx'][:100]
    noise_h5_out = hf['noise'][:100]
    fwhm_h5_out = hf['FWHM'][:100]
print(size_out)    

t1 = time()
runtimes['Combination of HDF5 files'] = calculate_runtime(t0,t1)    


t1_full = time()
runtimes['full script'] = calculate_runtime(t0_full,t1_full)
