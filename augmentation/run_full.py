# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020

@author: pielsticker
"""
from time import time
import os
import datetime
import h5py
from creator import Creator, calculate_runtime, check_db
t0 = time()
#%%
no_of_simulations = 500
no_of_files = 1000
input_filenames =  ['Fe_metal_Mark_shifted','FeO_Mark_shifted',
                    'Fe3O4_Mark_shifted','Fe2O3_Mark_shifted']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'iron_Mark_variable_linear_combination_no_broadening'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
os.makedirs(filepath)
filename_basic = os.path.join(*[filepath,time_and_run_name])

# Create multiple sets of similar spectra with the same settings
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_filenames, single = False)
    creator.run(broaden = True, x_shift = True, noise = True)
    creator.plot_random(1)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full')
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

#%%
no_of_simulations = 500
no_of_files = 1000
input_filenames =  ['Fe_metal','FeO','Fe3O4','Fe2O3']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'iron_variable_linear_combination_no_broadening'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
os.makedirs(filepath)
filename_basic = os.path.join(*[filepath,time_and_run_name])

# Create multiple sets of similar spectra with the same settings
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_filenames, single = False)
    creator.run(broaden = True, x_shift = True, noise = True)
    creator.plot_random(1)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full')
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

#%% Write to HDF5 files
from json_to_hdf5 import to_hdf5

output_datafolder = r'C:\Users\pielsticker\Simulations\\'
output_file = output_datafolder + '20200723_iron_Mark_variable_linear_combination_no_broadening.h5'
simulation_name = '20200723_iron_Mark_variable_linear_combination_no_broadening'
no_of_files_per_load = 50

to_hdf5(output_file, simulation_name, no_of_files_per_load)
print('finished saving')

with h5py.File(output_file, 'r') as hf:
    size = hf['X'].shape
    X_h5 = hf['X'][:4000,:,:]
    y_h5 = hf['y'][:4000,:]
    shiftx_h5 = hf['shiftx'][:4000,:]
    noise_h5 = hf['noise'][:4000,:]
    fwhm_h5 = hf['FWHM'][:4000,:]


output_datafolder = r'C:\Users\pielsticker\Simulations\\'
output_file = output_datafolder + '20200723_iron_variable_linear_combination_no_broadening.h5'
simulation_name = '20200723_iron_variable_linear_combination_no_broadening'
no_of_files_per_load = 50

to_hdf5(output_file, simulation_name, no_of_files_per_load)
print('finished saving')

with h5py.File(output_file, 'r') as hf:
    size = hf['X'].shape
    X_h5 = hf['X'][:4000,:,:]
    y_h5 = hf['y'][:4000,:]
    shiftx_h5 = hf['shiftx'][:4000,:]
    noise_h5 = hf['noise'][:4000,:]
    fwhm_h5 = hf['FWHM'][:4000,:]
    
#%%  Combine the datasets    
import numpy as np
from sklearn.utils import shuffle

def load_data(filenames):
    input_datafolder = r'C:\Users\pielsticker\Simulations'
    no_of_examples = 0
    for filename in filenames[:1]:
        input_filepath = os.path.join(input_datafolder,filename)
        with h5py.File(input_filepath, 'r') as hf:
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


filenames = ['20200723_iron_Mark_variable_linear_combination_no_broadening.h5',
             '20200723_iron_variable_linear_combination_no_broadening.h5']

X, y, shiftx, noise, fwhm, y_one, y_two = load_data(filenames)
X_shuff, y_shuff, shiftx_shuff, noise_shuff, fwhm_shuff = \
    shuffle(X, y, shiftx, noise, fwhm)

output_file = r'C:\Users\pielsticker\Simulations\20200723_iron_Mark_variable_linear_combination_no_broadening_combined_data.h5'   

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

t1 = time()
runtime = calculate_runtime(t0,t1)
print(f'Runtime: {runtime}.')