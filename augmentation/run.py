# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020

@author: pielsticker
"""
from time import time
import os
import datetime
from creator import Creator, calculate_runtime, check_db

no_of_simulations = 500
no_of_files = 200#1000
input_filenames =  ['Fe_metal','FeO',
                    'Fe3O4','Fe2O3']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'iron_variable_linear_combination_gas_phase'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
os.makedirs(filepath)
filename_basic = os.path.join(*[filepath,time_and_run_name])

#%% Create multiple sets of similar spectra with the same settings
t0 = time()
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_filenames, single = False )
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
runtime = calculate_runtime(t0,t1)
print(f'Runtime: {runtime}.')
