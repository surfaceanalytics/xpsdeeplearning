# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:29:41 2021

@author: pielsticker
"""
import os
import json
import csv

import tensorflow as tf
from tensorflow.keras import callbacks
#%%
class ExperimentLogging():
    def __init__(self, dir_name):
        self.root_dir = os.getcwd()
        self.model_dir = os.path.join(*[self.root_dir,
                                        'saved_models',
                                        dir_name])
        self.log_dir = os.path.join(*[self.root_dir,
                                      'logs',
                                      dir_name])
        self.fig_dir = os.path.join(*[self.root_dir,
                                      'figures',
                                      dir_name])
        
        ### Callbacks ###
        self.active_cbs = []
        
        # Save the model if a new best validation_loss is achieved.
        model_file_path = self.model_dir
        self.checkpoint_callback = callbacks.ModelCheckpoint(
                filepath = model_file_path,
                monitor = 'val_loss',
                verbose = 0,
                save_best_only = True ,
                save_weights_only = False,
                mode = 'auto',
                save_freq = 'epoch')

        # Stop the training if the validation loss increases for 
        # three epochs.
        self.es_callback = callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0, 
            patience = 3,
            verbose = True,
            mode = 'auto',
            baseline = None,
            restore_best_weights = True)
        
        # Log all training data in a format suitable
        # for TensorBoard anaylsis.
        self.tb_callback = callbacks.TensorBoard(
            log_dir = self.log_dir,
            histogram_freq = 1,
            write_graph = True,
            write_images = False,
            profile_batch = 0)
            
        # Log the results of each epoch in a csv file.
        csv_file = os.path.join(self.log_dir,'log.csv')
        self.csv_callback = callbacks.CSVLogger(
            filename = csv_file,
            separator=',',
            append = True)
        
        # Update saved hyperparameters JSON file.
        self.hp_callback = HyperParamCallback(self)
    
    def make_dirs(self):        
        if os.path.isdir(self.model_dir) == False:
            os.makedirs(self.model_dir)
            if os.path.isdir(self.model_dir) == True:
                print('Model folder created at ' +\
                      str(self.model_dir.split(self.root_dir)[1]))
        else:
            print('Model folder was already at ' +\
                  str(self.model_dir.split(self.root_dir)[1]))
        
        if os.path.isdir(self.log_dir) == False:
            os.makedirs(self.log_dir)
            if os.path.isdir(self.log_dir) == True:
                print('Logs folder created at ' +\
                      str(self.log_dir.split(self.root_dir)[1]))
        else:
            print('Logs folder was already at ' +\
                  str(self.log_dir.split(self.root_dir)[1]))
        
        if os.path.isdir(self.fig_dir) == False:
            os.makedirs(self.fig_dir)
            if os.path.isdir(self.fig_dir) == True:
                print('Figures folder created at ' +\
                      str(self.fig_dir.split(self.root_dir)[1]))
        else:
            print('Figures folder was already at ' +\
                  str(self.fig_dir.split(self.root_dir)[1]))
                
    def activate_cbs(self, 
                     checkpoint = True,
                     early_stopping = False,
                     tb_log = False, 
                     csv_log = True,
                     hyperparam_log = True):
        
        if checkpoint:
            self.active_cbs.append(self.checkpoint_callback)
        if early_stopping:
            self.active_cbs.append(self.es_callback)
        if tb_log:
            self.active_cbs.append(self.tb_callback)
        if csv_log:
            self.active_cbs.append(self.csv_callback)
        if hyperparam_log:
            self.active_cbs.append(self.hp_callback)

    def _count_epochs_trained(self):
        """
        Count the numbers of previously trained epochs by searching the 
        csv log file

        Returns
        -------
        epochs : int
            No. of epochs for which the model was trained.

        """
        csv_file = os.path.join(self.log_dir,'log.csv')
        epochs = 0
        
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    epochs += 1                    
        except:
            pass
        
        return epochs
    
    def _get_total_history(self, task):
        """
        Loads the previous training history from the CSV log file.
        Useful for retraining.

        Returns
        -------
        history : dict
            Dictionary containing the previous training history.

        """
        csv_file = os.path.join(self.log_dir,'log.csv')

        if task == 'classification':
            history = {'accuracy' : [],
                       'loss' : [],
                       'val_accuracy': [],
                       'val_loss': []}
            
            try:
                with open(csv_file, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        d = dict(row)
                        history['accuracy'].append(float(d['accuracy']))
                        history['loss'].append(float(d['loss']))
                        history['val_accuracy'].append(
                            float(d['val_accuracy']))
                        history['val_loss'].append(float(d['val_loss']))                
            except:
                pass
                    
        elif task == 'regression':
            history = {'loss' : [],
                       'val_loss': []}
            try:
                with open(csv_file, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        d = dict(row)
                        history['loss'].append(float(d['loss']))
                        history['val_loss'].append(float(d['val_loss']))
            except:
                pass
        
        return history
    

    def save_hyperparams(self):
        hyperparam_file_name = os.path.join(self.log_dir,
                                            'hyperparameters.json')
        
        with open(hyperparam_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(self.hyperparams,
                      json_file, 
                      ensure_ascii=False,
                      indent=4)

    def update_saved_hyperparams(self, new_params):
        for key in new_params:
            self.hyperparams[key] = new_params[key]
        self.save_hyperparams()        

class HyperParamCallback(callbacks.Callback):
    def __init__(self, logging):
        self.logging = logging
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_param = {
            'epochs_trained' : self.logging._count_epochs_trained()
            }  
        self.logging.update_saved_hyperparams(epoch_param)