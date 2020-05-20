# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020

@author: pielsticker
"""
from time import time
from creator import Creator, calculate_runtime, check_db
import json
import os

no_of_simulations = 1000
no_of_files = 200
input_labels =  ['Fe_metal','FeO','Fe3O4','Fe2O3']
filename_basic = '20200519_iron_single_'

#%% Create multiple sets of similar spectra with the same settings
t0 = time()
for i in range(0,10):
    for j in range(no_of_files):
        creator = Creator(no_of_simulations, input_labels, single = True)
        creator.run(broaden = True, x_shift = True, noise = True)
        #creator.plot_random(5)
        filename = filename_basic + str(2000+i*no_of_files+j)   
        #creator.upload_to_DB(filename, reduced = False)
        #collections = check_db(filename_i)
        #drop_db_collection(filename)
        creator.to_file(name = filename,
                        filetype = 'json',
                        how = 'full',
                        single = False)
        print('Finished set ' + str(i+1) + ' of ' + str(200))

t1 = time()
runtime = calculate_runtime(t0,t1)
print(f'Runtime: {runtime}.')
del(t0,t1,runtime,filename,i,j)



