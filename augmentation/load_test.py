# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:45:09 2020

@author: pielsticker
"""
import numpy as np
import os
os.chdir(r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\deepxps')
from xpsdeeplearning.network.data_handling import DataHandler

#%%
np.random.seed(502)
input_filepath = r'C:\Users\pielsticker\Simulations\20210222_Fe_linear_combination_small_gas_phase.h5'

datahandler = DataHandler(intensity_only = False)
train_test_split = 0.2
train_val_split = 0.2
no_of_examples = 180
    
X_train, X_val, X_test, y_train, y_val, y_test,\
    aug_values_train, aug_values_val, aug_values_test =\
        datahandler.load_data_preprocess(
            input_filepath = input_filepath,
            no_of_examples = no_of_examples,
            train_test_split = train_test_split,
            train_val_split = train_val_split)        
print('Input shape: ' + str(datahandler.input_shape))
print('Labels: ' + str(datahandler.labels))
print('No. of classes: ' + str(datahandler.num_classes))
  
datahandler.plot_random(no_of_spectra = 9,
                        dataset = 'train',
                        with_prediction = False)
            

            
            