# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020

@author: pielsticker

This script is used to simulate many spectra using the Creator class
and save the data to JSON files. The no_of_simulations parameter is 
used to control the number of spectra stored in each JSON file and the
no_of_files parameter determines how many JSON files are created.

Total no. of spectra = no_of_simulations*no_of_files 
""" 

from time import time
import os
import json
import datetime
import h5py
from creator import Creator, calculate_runtime
from json_to_hdf5 import to_hdf5

#%% Input parameter
### Change the following line according to your folder structure ###
init_param_folder = r'C:\Users\pielsticker\Simulations'

#%% Parameters
init_param_filepath = os.path.join(init_param_folder,
                                   'init_params.json')

with open(init_param_filepath, 'r') as param_file:
    run_params = json.load(param_file)

no_of_files = run_params['no_of_files']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
time_and_run_name = timestamp + '_' + run_params['run_name']
output_datafolder = run_params['output_datafolder']
json_datafolder = os.path.join(*[output_datafolder,time_and_run_name])

try:
    os.makedirs(json_datafolder)
except:
    pass

filename_basic = os.path.join(*[json_datafolder,time_and_run_name])

#%% Create multiple sets of similar spectra with the same settings
runtimes = {}

t0 = time()
for i in range(no_of_files):
    creator = Creator(run_params)
    creator.run()
    creator.plot_random(5)
    filename = filename_basic + str(i)   
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full')
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

run_params['timestamp'] = timestamp
run_params['h5_filename'] = time_and_run_name + '.h5'

with open(os.path.join(json_datafolder, 'run_params.json'), 'w') as out_file:
    json.dump(run_params, out_file, indent=4)

t1 = time()
runtime = calculate_runtime(t0,t1)
runtimes['dataset_creation'] = runtime
print(f'Runtime: {runtime}.')


#%% Save all to HDF5.
t0 = time()
h5_filepath = run_params['output_datafolder'] + run_params['h5_filename']
to_hdf5(json_datafolder,
        h5_filepath)
t1 = time()
runtimes['h5_save'] = calculate_runtime(t0,t1)
print('finished saving')

# Test new file.
t0 = time()    
with h5py.File(h5_filepath, 'r') as hf:
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