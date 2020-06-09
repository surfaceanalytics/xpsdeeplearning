# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:03:47 2020

@author: pielsticker
"""
import os
import numpy as np
import json
import h5py
from time import time

from creator import calculate_runtime



def load_data_preprocess(input_datafolder,start,end):
    label_values = ['Fe metal','FeO','Fe3O4','Fe2O3']
    
    filenames = next(os.walk(input_datafolder))[2]

    X = []
    y = []
        
    for file in filenames[start:end]:
        filename = input_datafolder + file
        with open(filename, 'r') as json_file:
            test = json.load(json_file)
        for j in range(0,len(test)):
            X_one = test[j]['y']
            y_one = test[j]['label']
            X.append(X_one)
            y.append(y_one)
        print('Load: ' + str((filenames.index(file)-start)*len(test)+j+1) + '/' + \
                      str(len(filenames[start:end])*len(test)))
                                       
    y = _one_hot_encode(y, label_values)
                
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.array(y)
        
    return X, y
        
    
def _one_hot_encode(y, label_values):
    labels = [i for s in [d.keys() for d in y] for i in s]
    new_labels = []
        
    for label in labels:  
        number = [i for i,x in enumerate(label_values) if x == label][0]
        label_list = [0,0,0,0]
        label_list[number] = 1
        new_labels.append(label_list)
                        
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
        X, y  = load_data_preprocess(input_datafolder, start, end)
        hf.create_dataset('X', data = X,
                          compression="gzip", chunks=True,
                          maxshape=(None,X.shape[1],X.shape[2]))        
        hf.create_dataset('y', data = y,
                          compression="gzip", chunks=True,
                          maxshape=(None, y.shape[1]))
        print('Saved: ' + str(1) + '/' + str(no_of_loads))
        for load in range(1,no_of_loads):
            start = load*no_of_files_per_load
            end = start+no_of_files_per_load
            X_new, y_new  = load_data_preprocess(input_datafolder, start, end)
            hf["X"].resize((hf["X"].shape[0] + X_new.shape[0]), axis = 0)
            hf["X"][-X_new.shape[0]:] = X_new
            hf["y"].resize((hf["y"].shape[0] + y_new.shape[0]), axis = 0)
            hf["y"][-X_new.shape[0]:] = y_new
    
            print('Saved: ' + str(load+1) + '/' + str(no_of_loads))



#%%               
if __name__ == "__main__":
    input_datafolder = r'C:\Users\pielsticker\Simulations\20200605_iron_single'
    output_datafolder = r'C:\Users\pielsticker\Simulations\\'
    output_file = output_datafolder + '20200605_iron_single.h5'
    simulation_name = '20200605_iron_single'
    no_of_files_per_load = 100

    runtimes = {}
    t0 = time()
    to_hdf5(output_file, simulation_name, no_of_files_per_load)
    t1 = time()
    runtimes['h5_save_iron_single'] = calculate_runtime(t0,t1)
    print('finished saving')
    
    t0 = time()
    with h5py.File(output_file, 'r') as hf:
        X_h5_full = hf['X']
        x_h5 = X_h5_full[:50000,:,:]
        y_h5_full = hf['y']
        y_h5 = y_h5_full[:50000,:]
        t1 = time()
        runtimes['h5_load_iron_single'] = calculate_runtime(t0,t1)

    
