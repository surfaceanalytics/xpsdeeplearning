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
import datetime
from creator import Creator, calculate_runtime, check_db

#%% Parameters
no_of_simulations = 500
no_of_files = 500
input_filenames =  ['Pd_metal_narrow','PdO_narrow']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'palladium_linear_combination_gas_phase'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
try:
    os.makedirs(filepath)
except:
    pass
filename_basic = os.path.join(*[filepath,time_and_run_name])

#%% Create multiple sets of similar spectra with the same settings
t0 = time()
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_filenames, single = False )
    creator.run(broaden = False, x_shift = True, noise = True, scatter = True)
    creator.plot_random(5)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full')
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

t1 = time()
runtime = calculate_runtime(t0,t1)
print(f'Runtime: {runtime}.')
