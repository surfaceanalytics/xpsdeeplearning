# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:00 2020

@author: pielsticker
"""
from time import time
from creator import Creator, calculate_runtime, check_db

no_of_simulations = 1000
no_of_files = 4000
input_labels =  ['Fe_metal','FeO','Fe3O4','Fe2O3']
filename_basic = '20200528_iron_single_x_shift_'

#%% Create multiple sets of similar spectra with the same settings
t0 = time()
for i in range(no_of_files):
    creator = Creator(no_of_simulations, input_labels, single = True)
    creator.run(broaden = False, x_shift = True, noise = False)
    creator.plot_random(1)
    filename = filename_basic + str(i)   
    #creator.upload_to_DB(filename, reduced = False)
    #collections = check_db(filename)
    #drop_db_collection(filename)
    creator.to_file(name = filename,
                    filetype = 'json',
                    how = 'full',
                    single = False)
    print('Finished set ' + str(i+1) + ' of ' + str(no_of_files))

t1 = time()
runtime = calculate_runtime(t0,t1)
print(f'Runtime: {runtime}.')
del(t0,t1,runtime,filename,i)



