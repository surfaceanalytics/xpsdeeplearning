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
no_of_files = 1000
input_labels =  ['Fe_metal','FeO','Fe3O4','Fe2O3']
timestamp = datetime.datetime.now().strftime("%Y%m%d")
run_name = 'iron_variable_linear_combination'
time_and_run_name = timestamp + '_' + run_name

datafolder = r'C:\Users\pielsticker\Simulations'
filepath = os.path.join(*[datafolder,time_and_run_name])
os.makedirs(filepath)
filename_basic = os.path.join(*[filepath,time_and_run_name])

#%% Create multiple sets of similar spectra with the same settings
t0 = time()
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_labels, single = False)
    creator.run(broaden = True, x_shift = True, noise = True)
    creator.plot_random(1)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(filepath = filename,
                    filetype = 'json',
                    how = 'full',
                    single = False)
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

t1 = time()
runtime = calculate_runtime(t0,t1)
print(f'Runtime: {runtime}.')