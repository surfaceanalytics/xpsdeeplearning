# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:55:44 2020

@author: pielsticker
"""

import os
import shelve
import numpy as np
import json
import csv
import h5py
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping,\
    ModelCheckpoint, TensorBoard, CSVLogger

from .models import CustomModel, CustomModelSimpleCNN,\
    CustomModelCNN, CustomModelMLP
    
#%%
class Classifier():
    def __init__(self, time, model_type ='CNN', model_name = 'Classifier', 
                 labels = []):
        self.time = time
        self.model_name = model_name
        self.model_type = model_type
        self.label_values = labels
        
        self.num_classes = len(self.label_values)
        
        root_dir = os.getcwd()
        dir_name = self.time + '_' + self.model_name 
        
        self.model_dir = os.path.join(*[root_dir, 'saved_models', dir_name])
        self.log_dir = os.path.join(*[root_dir, 'logs', dir_name])
        self.fig_dir = os.path.join(*[root_dir, 'figures', dir_name])
        
        for item in [self.model_dir, self.log_dir, self.fig_dir]:
            if os.path.isdir(item) == False:
                os.makedirs(item)
        
        if os.path.isdir(self.model_dir) == True:
            print('Model folder created at ' +\
                  str(self.model_dir.split(root_dir)[1]))
        if os.path.isdir(self.log_dir) == True:
            print('Log folder created at ' +\
                  str(self.log_dir.split(root_dir)[1]))
        if os.path.isdir(self.fig_dir) == True:
            print('Figure folder created at ' +\
                  str(self.fig_dir.split(root_dir)[1]))
                
                
    def load_data_preprocess(self, input_filepath, no_of_examples,
                             train_test_split, train_val_split):
        self.input_filepath = input_filepath
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.no_of_examples = no_of_examples

        with h5py.File(input_filepath, 'r') as hf:
            dataset_size = hf['X'].shape[0]
            r = np.random.randint(0, dataset_size-self.no_of_examples)
            X = hf['X'][r:r+self.no_of_examples, :, :]
            y = hf['y'][r:r+self.no_of_examples, :]
        
        # Shuffle X and y together
        self.X, self.y = shuffle(X, y)

        # Split into train, val and test sets
        self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = \
                self._split_test_val_train(self.X, self.y)
                
        self.input_shape = (self.X_train.shape[1], 1)
   
        return self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test
    
    
    def _split_test_val_train(self, X, y):         
        # First split into train+val and test sets
        no_of_train_val = int((1-self.train_test_split) *\
                              X.shape[0])
        
        X_train_val = X[:no_of_train_val,:,:]
        X_test = X[no_of_train_val:,:,:]
        y_train_val = y[:no_of_train_val,:]
        y_test = y[no_of_train_val:,:]
        
        # Then create val subset from train set
        no_of_train = int((1-self.train_val_split) *\
                          X_train_val.shape[0])
                    
        X_train = X_train_val[:no_of_train,:,:]
        X_val = X_train_val[no_of_train:,:,:]
        y_train = y_train_val[:no_of_train,:]
        y_val = y_train_val[no_of_train:,:]
        
        print("Data was loaded!")
        print('Total no. of samples: ' + str(self.X.shape[0]))
        print('No. of training samples: ' + str(X_train.shape[0]))
        print('No. of validation samples: ' + str(X_val.shape[0]))
        print('No. of test samples: ' + str(X_test.shape[0]))
        print('Shape of each sample : '
              + str(X_train.shape[1]) + ' features (X)' 
              + ' + ' + str(y_train.shape[1])
              + ' labels (y)')
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def check_class_distribution(self):
        class_distribution = {'all data': {},
                              'training data': {},
                              'validation data': {},
                              'test data': {}}
        
        for i in range(self.num_classes):
            class_distribution['all data'][str(i)] = 0
            class_distribution['training data'][str(i)] = 0
            class_distribution['validation data'][str(i)] = 0
            class_distribution['test data'][str(i)] = 0
             
        data_list = [self.y, self.y_train, self.y_val,self.y_test]
       
        for i, dataset in enumerate(data_list):
            key = list(class_distribution.keys())[i]
            
            for j in range(dataset.shape[0]): 
                argmax_class = np.argmax(dataset[j,:], axis = 0)
                class_distribution[key][str(argmax_class)] +=1
                
        self.class_distribution = class_distribution
                
        return self.class_distribution
        
    
    def plot_class_distribution(self):               
        data = []
        for k, v in self.class_distribution.items():
            data_list = []
            for key, value in v.items():
                data_list.append(value)
            data.append(data_list)
        data = np.transpose(np.array(data))
            
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        x = np.arange(len(self.class_distribution.keys()))*1.5
 
        for i in range(data.shape[0]):
            ax.bar(x + i*0.25, data[i], align='edge', width = 0.2)
        plt.title('Class distribution')
        plt.legend(self.label_values)   
        plt.xticks(ticks=x+.5, labels=list(self.class_distribution.keys()))
        plt.show()
        
        
    def plot_random(self, no_of_spectra, dataset = 'train',
                    with_prediction = False): 
        no_of_rows = int(no_of_spectra/3)
        no_of_cols = 3
        if (no_of_spectra % no_of_cols) != 0:
            no_of_rows += 1
            
        fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
        plt.subplots_adjust(left=0.125, bottom=0.5, right=2.7,
                            top=no_of_rows, wspace=0.2, hspace=0.2)
    
        for i in range(no_of_spectra):
            x = np.arange(694, 750.05, 0.05)

            if dataset == 'train':
                r = np.random.randint(0, self.X_train.shape[0])
                y = self.X_train[r]
                labels = self.y_train[r]
                if with_prediction == True:
                    real_y = ('Real: ' + \
                              str(self.y_train[r]) + '\n')
                    # Round prediction and sum to 1
                    tmp_array = np.around(self.pred_train[r], decimals = 4)
                    row_sums = tmp_array.sum()
                    tmp_array = tmp_array / row_sums
                    tmp_array = np.around(tmp_array, decimals = 3)    
                    pred_y = ('Prediction: ' +\
                              str(tmp_array) + '\n')
                    pred_label = ('Predicted label: ' +\
                                  str(self.pred_train_classes[r,0]))
                    pred = real_y + pred_y + pred_label

            elif dataset == 'val':
                r = np.random.randint(0, self.X_val.shape[0])
                y = self.X_val[r]
                labels = self.y_val[r]
                
            elif dataset == 'test':
                r = np.random.randint(0, self.X_test.shape[0])
                y = self.X_test[r]
                labels = self.y_test[r]
                if with_prediction == True:
                    real_y = ('Real: ' + \
                              str(self.y_test[r]) + '\n')
                    # Round prediction and sum to 1
                    tmp_array = np.around(self.pred_test[r], decimals = 4)
                    row_sums = tmp_array.sum()
                    tmp_array = tmp_array / row_sums
                    tmp_array = np.around(tmp_array, decimals = 3)    
                    pred_y = ('Prediction: ' +\
                              str(tmp_array) + '\n')
                    pred_label = ('Predicted label: ' +\
                                  str(self.pred_test_classes[r,0]))
                    pred = real_y + pred_y + pred_label
            
            for j, value in enumerate(labels):
                if value == 1:
                    label = str(self.label_values[j])
                    if with_prediction == True:
                        label =  ('Real label: ' + label)
                    
            row, col = int(i/no_of_cols), i % no_of_cols
            axs[row, col].plot(np.flip(x),y)
            axs[row, col].invert_xaxis()
            axs[row, col].set_xlim(750.05,694)
            axs[row, col].set_xlabel('Binding energy (eV)')
            axs[row, col].set_ylabel('Intensity (arb. units)')                          
            axs[row, col].text(0.025, 0.9, label,
                               horizontalalignment='left',
                               verticalalignment='top',
                               transform = axs[row, col].transAxes,
                               fontsize = 12) 
            if with_prediction == True:
                axs[row, col].text(0.025, 0.3, pred,
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   transform = axs[row, col].transAxes,
                                   fontsize = 12) 

    
    def build_model(self):        
        if self.model_type == 'CNN_simple':
            self.model = CustomModelSimpleCNN(self.input_shape,
                                              self.num_classes)
            
        elif self.model_type == 'CNN':
            self.model = CustomModelCNN(self.input_shape,
                                        self.num_classes)
            
        elif self.model_type == 'MLP':
            self.model = CustomModelMLP(self.input_shape,
                                        self.num_classes)
            
    
    def summary(self):
        print(self.model.summary())
        
        #Save model summary to a string
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)
        
        print("Model created!")
    
    
    def save_and_print_model_image(self):        
        fig_file_name = os.path.join(self.fig_dir,'model.png')
        plot_model(self.model, to_file = fig_file_name,
                   rankdir = "LR", show_shapes = True,
                   show_layer_names = True,
                   expand_nested = False,
                   dpi = 150)
        model_plot = plt.imread(fig_file_name)
        fig, ax = plt.subplots(figsize=(18, 2))
        ax.imshow(model_plot, interpolation='nearest')
        plt.tight_layout()
        plt.show()
        

    def train(self, checkpoint = True, early_stopping = False,
              tb_log = False, csv_log = True, epochs = 200, batch_size = 32,
              verbose = 1):
        self.epochs = epochs
        self.batch_size = batch_size
        epochs_trained = self._count_epochs_trained()
        
        callbacks = []
        
        if checkpoint:
            model_file_name = self.model_dir
            checkpoint_callback = ModelCheckpoint(filepath = model_file_name,
                                                  monitor = 'val_loss',
                                                  verbose = 0,
                                                  save_best_only = True ,
                                                  save_weights_only = False,
                                                  mode = 'auto',
                                                  save_freq = 'epoch')
            callbacks.append(checkpoint_callback)


        if early_stopping:
            es_callback = EarlyStopping(monitor = 'val_loss',
                                        min_delta = 0, 
                                        patience = 3,
                                        verbose = True,
                                        mode = 'auto',
                                        baseline = None,
                                        restore_best_weights = True)
            callbacks.append(es_callback)
        
        if tb_log:            
            tb_callback = TensorBoard(log_dir = self.log_dir,
                                      histogram_freq = 1,
                                      write_graph = True,
                                      write_images = False)
            callbacks.append(tb_callback)
            
        if csv_log:            
            csv_file = os.path.join(self.log_dir,'log.csv')
            
            csv_callback = CSVLogger(filename = csv_file,
                                      separator=',',
                                      append = True)
            callbacks.append(csv_callback)
            
        try:
            training = self.model.fit(self.X_train,
                                      self.y_train,
                                      validation_data = \
                                          (self.X_val, self.y_val),
                                          epochs = self.epochs +\
                                              epochs_trained,
                                      batch_size = self.batch_size,
                                      initial_epoch = epochs_trained,
                                      verbose = verbose,
                                      callbacks = callbacks) 
            
            self.last_training = training.history
            self.history = self._get_total_history()
            print('Training done!')
           
        except KeyboardInterrupt:
            self.save_model()
            self.history = self._get_total_history()
            print('Training interrupted!')
        
        return self.history
        
    
    def evaluate(self):
        self.score = self.model.evaluate(self.X_test,
                                   self.y_test,
                                   batch_size = self.batch_size,
                                   verbose=True)      
        print('Evaluation done! \n')
        
        self.test_loss, self.test_accuracy = self.score[0], self.score[1]
        
        return self.score
     
     
    def predict(self, verbose = True):
        self.pred_train = self.model.predict(
            self.X_train, verbose = verbose)
        self.pred_test = self.model.predict(
            self.X_test, verbose = verbose)
        
        if verbose == True:
            print('Prediction done!')
        
        return self.pred_train, self.pred_test
    
    
    def predict_classes(self):
        pred_train, pred_test = self.predict(verbose = False)
        
        pred_train_classes = []
        pred_test_classes = []
        
        for i in range(pred_train.shape[0]): 
            argmax_class = np.argmax(pred_train[i,:], axis = 0)
            pred_train_classes.append(self.label_values[argmax_class])
            
        for i in range(pred_test.shape[0]): 
            argmax_class = np.argmax(pred_test[i,:], axis = 0)
            pred_test_classes.append(self.label_values[argmax_class])
            
        self.pred_train_classes = np.array(pred_train_classes).reshape(-1,1)
        self.pred_test_classes = np.array(pred_test_classes).reshape(-1,1)
        
        print('Class prediction done!')
        
        return self.pred_train_classes, self.pred_test_classes
    
    def plot_wrong_classification(self):
        binding_energy = np.arange(694, 750.05, 0.05)
        
        no_of_wrong_pred = -1
        wrong_pred_args = []
        
        for i in range(self.pred_test.shape[0]): 
            argmax_class_true = np.argmax(self.y_test[i,:], axis = 0)
            argmax_class_pred = np.argmax(self.pred_test[i,:], axis = 0)
            
        if argmax_class_true != argmax_class_pred:
            no_of_wrong_pred += 1
            wrong_pred_args.append(i)

        no_of_rows = int(no_of_wrong_pred/3)
        no_of_cols = 3
        if (no_of_wrong_pred % no_of_cols) != 0:
            no_of_rows += 1

        fig, axs = plt.subplots(nrows = no_of_rows, ncols = no_of_cols)
        plt.subplots_adjust(left=0.125, bottom=0.5, right=2.7,
                        top=no_of_rows, wspace=0.2, hspace=0.2)
        
        for n in range(no_of_wrong_pred):
            arg = wrong_pred_args[n]
            intensity = self.X_test[arg]
            
            real_y = ('Real: ' + \
                str(self.y_test[arg]) + '\n')
            # Round prediction and sum to 1
            tmp_array = np.around(self.pred_test[arg], decimals = 4)
            row_sums = tmp_array.sum()
            tmp_array = tmp_array / row_sums
            tmp_array = np.around(tmp_array, decimals = 2)
            pred_y = ('Prediction: ' +\
                      str(tmp_array) + '\n')
            pred_label = ('Predicted label: ' +\
                          str(self.pred_test_classes[n,0]))
            labels = self.y_test[arg]
            for j, value in enumerate(labels):
                if value == 1:
                    label = str(self.label_values[j])
                    label =  ('Real label: ' + label + '\n')
                
            text = real_y + pred_y + label + pred_label
            
            row, col = int(n/no_of_cols), n % no_of_cols
            axs[row, col].plot(np.flip(binding_energy),intensity)
            axs[row, col].invert_xaxis()
            axs[row, col].set_xlim(750.05,694)
            axs[row, col].set_xlabel('Binding energy (eV)')
            axs[row, col].set_ylabel('Intensity (arb. units)')  
            axs[row, col].text(0.025, 0.35, text,
                               horizontalalignment='left',
                               verticalalignment='top',
                               transform = axs[row, col].transAxes,
                               fontsize = 12)
            
    
    def save_model(self):        
        model_file_name = os.path.join(self.model_dir,
                                       'model.json')
        hyperparam_file_name = os.path.join(self.model_dir,
                                            'hyperparameters.json')
        weights_file_name = os.path.join(self.model_dir,
                                         'weights.h5')
        
        model_json = self.model.to_json()
        with open(model_file_name, 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_file_name)
        self.model.save(self.model_dir)

        params_dict = {
            'model_name' : self.model_name,
            'datetime' : self.time,
            'Input file' : self.input_filepath,
            'model_summary' : self.model_summary,
            'num_of_classes' : self.num_classes,
            'train_test_split_percentage' : self.train_test_split,
            'train_val_split_percentage' : self.train_val_split,
            'epochs_trained' : self._count_epochs_trained(),
            'batch_size' : self.batch_size,
            'class_distribution' : self.check_class_distribution(),
            'loss' : 'Categorical Cross-Entropy',
            'optimizer' : self.model.opt._name,
            'learning rate' : 0.00001,
            'Labels': self.label_values,
            'Total no. of samples' : str(self.X.shape[0]),
            'No. of training samples' : str(self.X_train.shape[0]),
            'No. of validation samples' : str(self.X_val.shape[0]),
            'No. of test samples' : str(self.X_test.shape[0]),
            'Shape of each sample' : str(self.X_train.shape[1]) + \
                ' features (X)' + ' + ' + str(self.y_train.shape[1])
                + ' labels (y)'}
            
        with open(hyperparam_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(params_dict, json_file, ensure_ascii=False, indent=4)
        print("Saved model and hyperparameters.")
        
    
    def shelve_results(self, full = False):
        filename = os.path.join(self.model_dir,'vars')
        
        with shelve.open(filename,'n') as shelf:
            key_list = ['y_train', 'y_test', 'pred_train', 'pred_test',
                    'pred_train_classes', 'pred_test_classes',
                    'test_accuracy', 'test_loss']
            if full == True:
                key_list.extend(['X, X_train', 'X_val', 'X_test', 'y',
                                 'y_val', 'class_distribution', 'hist'])
            for key in key_list:
                shelf[key] = vars(self)[key]
            
            
    def load_model(self, from_path = False, **kwargs):
        if (from_path == True) and ('model_path' in kwargs.keys()):
            file_name = kwargs['model_path']
        else:
            file_name = self.model_dir
 
        custom_objects = {'CustomModel': CustomModel}
        if self.model_type == 'CNN_simple':
            custom_objects['CustomModelSimpleCNN'] = CustomModelSimpleCNN
        
        elif self.model_type == 'CNN':
            custom_objects['CustomModelCNN'] = CustomModelCNN
        
        elif self.model_type == 'MLP':
            custom_objects['CustomModelMLP'] = CustomModelMLP
        
        self.model = load_model(file_name, custom_objects = custom_objects)
        print("Loaded model from disk.")
        
        
    def _count_epochs_trained(self):
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
        
    
    def _get_total_history(self):
        csv_file = os.path.join(self.log_dir,'log.csv')
        history = {'accuracy' : [],
                   'loss' : [],
                   'val_accuracy': [],
                   'val_loss': []}
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d = dict(row)
                    history['accuracy'].append(d['accuracy'])
                    history['loss'].append(d['loss'])
                    history['val_accuracy'].append(d['val_accuracy'])
                    history['val_loss'].append(d['val_loss'])                
        except:
            pass
        
        return history
