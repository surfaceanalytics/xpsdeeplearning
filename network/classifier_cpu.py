# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:58:25 2020

@author: pielsticker
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:16:51 2020

@author: pielsticker
"""


import os
import shelve
import numpy as np
import json
import csv
import datetime
import h5py
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

# Run tensorflow on local CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model, model_to_dot
from tensorflow.keras.callbacks import EarlyStopping,\
    ModelCheckpoint, TensorBoard, CSVLogger

from models import CustomModel, CustomModelSimpleCNN,\
    CustomModelCNN, CustomModelMLP
from utils import TrainingGraphs, SpectrumFigure, Report

#%%
class Classifier():
    def __init__(self, time, model_type ='CNN', model_name = 'Classifier'):
        self.time = time
        self.model_name = model_name
        self.model_type = model_type
        
        root_dir = os.getcwd().partition('network')[0]
        dir_name = self.time + '_' + self.model_name 
        self.model_dir = root_dir + '\\saved_models\\' + dir_name + '\\'
        self.log_dir = root_dir + '\\logs\\' + dir_name + '\\'
        self.fig_dir = root_dir + '\\figures\\' + dir_name + '\\'
        
        for item in [self.model_dir, self.log_dir, self.fig_dir]:
            if os.path.isdir(item) == False:
                os.makedirs(item)
        
        self.epochs = 2
        self.batch_size = 32
        self.train_test_split_percentage = 0.2
        self.train_val_split_percentage = 0.2
        self.no_of_examples = 100#000
        
        # Determine which dataset t shall be used
        self.input_datafolder = r'C:\Users\pielsticker\Simulations\\'                    
        self.input_file = '20200605_iron_single.h5'
        self.label_values = ['Fe metal', 'FeO', 'Fe3O4', 'Fe2O3']
        self.num_classes = len(self.label_values)
        
        # load dataset and preprocess it
        self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = \
                    self.load_data_preprocess()
                    
        self.input_shape = (self.X_train.shape[1], 1)
        
        self.build_model()
        
        self.summary()
        self.save_and_print_model_image()
        
        
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
        
    
    def load_data_preprocess(self):
        filename_load = self.input_datafolder + self.input_file

        with h5py.File(filename_load, 'r') as hf:
            dataset_size = hf['X'].shape[0]
            r = np.random.randint(0, dataset_size-self.no_of_examples)
            X = hf['X'][r:r+self.no_of_examples, :, :]
            y = hf['y'][r:r+self.no_of_examples, :]
        
        self.X, self.y = shuffle(X, y)
# =============================================================================
#         # Use only specific data
#         a = np.array([[1,0,0,0],[0,0,0,1]], dtype=np.float32)
#         data_filter = np.where(
#             (np.all(self.y==a[0],axis=1)) | (np.all(self.y==a[1],axis=1)))
#         self.X, self.y = self.X[data_filter],self.y[data_filter]
# =============================================================================

        # Split into train, val and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self._split_test_val_train(self.X, self.y)
   
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def _split_test_val_train(self, X, y):         
        # First split into train+val and test sets
        no_of_train_val = int((1-self.train_test_split_percentage) *\
                              X.shape[0])
        
        X_train_val = X[:no_of_train_val,:,:]
        X_test = X[no_of_train_val:,:,:]
        y_train_val = y[:no_of_train_val,:]
        y_test = y[no_of_train_val:,:]
        
        # Then create val subset from train set
        no_of_train = int((1-self.train_val_split_percentage) *\
                          X_train_val.shape[0])
                    
        X_train = X_train_val[:no_of_train,:,:]
        X_val = X_train_val[no_of_train:,:,:]
        y_train = y_train_val[:no_of_train,:]
        y_val = y_train_val[no_of_train:,:]
        
        print('\n')
        print("Data was loaded!")
        print('Total no. of samples: ' + str(self.X.shape[0]))
        print('No. of training samples: ' + str(X_train.shape[0]))
        print('No. of validation samples: ' + str(X_val.shape[0]))
        print('No. of test samples: ' + str(X_test.shape[0]))
        print('Shape of each sample : '
              + str(X_train.shape[1]) + ' features (X)' 
              + ' + ' + str(y_train.shape[1])
              + ' labels (y)\n')
        
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
                
        return class_distribution
        

    def train(self, checkpoint = True, early_stopping = False,
              tb_log = False, csv_log = True):
        epochs_trained = self.count_epochs_trained()
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
            csv_file = self.log_dir + 'log.csv'
            
            csv_callback = CSVLogger(filename = csv_file,
                                      separator=',',
                                      append = True)
            callbacks.append(csv_callback)
            
        try:
            training = self.model.fit(self.X_train,
                                      self.y_train,
                                      validation_data = \
                                          (self.X_val, self.y_val),
                                          nb_epoch = self.epochs +\
                                              epochs_trained,
                                      batch_size = self.batch_size,
                                      initial_epoch = epochs_trained,
                                      verbose = True,
                                      callbacks = callbacks) 
            
            self.last_training = training.history
            self.history = self.get_total_history()
            print("Training done!")
           
        except KeyboardInterrupt:
            self.save_model()
            self.history = self.get_total_history()
            print('Training interrupted!')
        
        return self.history
        
    
    def evaluate(self):
        loss = self.model.evaluate(self.X_test,
                                   self.y_test,
                                   batch_size = self.batch_size,
                                   verbose=True)      
        print("Evaluation done!")
        
        return loss
     
     
    def predict(self):
        prediction_training = self.model.predict(self.X_train, verbose = True)
        prediction_test = self.model.predict(self.X_test, verbose = True)
        
        print("Prediction done!")
        
        return prediction_training, prediction_test
    
    
    def predict_classes(self):
        pred_training, pred_test = self.predict()
        
        pred_training_classes = []
        pred_test_classes = []
        
        for i in range(pred_training.shape[0]): 
            argmax_class = np.argmax(pred_training[i,:], axis = 0)
            pred_training_classes.append(self.label_values[argmax_class])
            
        for i in range(pred_test.shape[0]): 
            argmax_class = np.argmax(pred_test[i,:], axis = 0)
            pred_test_classes.append(self.label_values[argmax_class])
            
        pred_training_classes = np.array(pred_training_classes).reshape(-1,1)
        pred_test_classes = np.array(pred_test_classes).reshape(-1,1)
        
        print("Class prediction done!")
        
        return pred_training_classes, pred_test_classes
            
    
    def save_model(self):        
        model_file_name = self.model_dir + 'model.json'
        hyperparam_file_name = self.model_dir + 'hyperparameters.json'
        weights_file_name = self.model_dir + "weights.h5"
        
        model_json = self.model.to_json()
        with open(model_file_name, 'w', encoding='utf-8') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_file_name)
        self.model.save(self.model_dir)
        #del self.model

        params_dict = {
            'model_name' : self.model_name,
            'datetime' : self.time,
            'Input file' : self.input_file,
            'model_summary' : self.model_summary,
            'num_of_classes' : self.num_classes,
            'train_test_split_percentage' : self.train_test_split_percentage,
            'train_val_split_percentage' : self.train_val_split_percentage,
            'epochs_trained' : self.count_epochs_trained(),
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
            
            
    def load_model(self, from_path = False, **kwargs):
        if (from_path == True) and ('model_path' in kwargs.keys()):
            file_name = kwargs['model_path']
        else:
            file_name = self.model_dir + 'model'
 
        custom_objects = {'CustomModel': CustomModel}
        if self.model_type == 'CNN_simple':
            custom_objects['CustomModelSimpleCNN'] = CustomModelSimpleCNN
        
        elif self.model_type == 'CNN':
            custom_objects['CustomModelCNN'] = CustomModelCNN
        
        elif self.model_type == 'MLP':
            custom_objects['CustomModelMLP'] = CustomModelMLP
        
        self.model = load_model(file_name, custom_objects = custom_objects)
        print("Loaded model from disk.")

        
    def plot_random(self, no_of_spectra):        
        for i in range(no_of_spectra):
            r = np.random.randint(0, self.X_train.shape[0])
            y = self.X_train[r]
            label = str(self.y_train[r])
            SpectrumFigure(y, label)
            plt.show()
            
    
    def summary(self):
        print(self.model.summary())
        
        #Save model summary to a string
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)
        
        fig_file_name = self.fig_dir + 'model_summary.jpeg'
        with open(fig_file_name, 'w') as f:
            print(self.model_summary, file=f)

        print("Model created!")
    
    
    def save_and_print_model_image(self):        
        fig_file_name = self.fig_dir + 'model.png'
        plot_model(self.model, to_file = fig_file_name,
                   rankdir = "LR", show_shapes = True,
                   show_layer_names = True,
                   expand_nested = False,
                   dpi = 150)
        model_plot = plt.imread(fig_file_name)
        plt.imshow(model_plot)
        plt.show()
        
        
    def count_epochs_trained(self):
        csv_file = self.log_dir + 'log.csv'
        epochs = 0
        
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    epochs += 1                    
        except:
            pass
        
        return epochs
        
    
    def get_total_history(self):
        csv_file = self.log_dir + 'log.csv'
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
    
    
    def shelve_results(self, full = False):
        filename = self.model_dir + '\\vars'  
        
        with shelve.open(filename,'n') as shelf:
            key_list = ['y_train', 'y_test', 'pred_train', 'pred_test',
                    'pred_train_classes', 'pred_test_classes',
                    'test_accuracy', 'test_loss']
            if full == True:
                key_list.extend(['X, X_train', 'X_val', 'X_test', 'y',
                                 'y_val', 'class_distribution', 'hist'])
            for key in key_list:
                shelf[key] = globals()[key]
# =============================================================================
#             try:
#                 shelf[key] = globals()[key]
#             except:
#                 # __builtins__, shelf, and imported modules can not
#                 # be shelved.
#                 print('ERROR shelving: {0}'.format(key))
# =============================================================================
    


#%% 
if __name__ == "__main__":
    np.random.seed(502)
    time =  datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm")
    model_type = 'CNN_simple'
    model_name = 'Fe_single_4_classes_CNN_simple'
    
    classifier = Classifier(time = time,
                            model_type = model_type,
                            model_name = model_name)
    
    class_distribution = classifier.check_class_distribution()
   
    X = classifier.X
    X_train = classifier.X_train
    X_val = classifier.X_val
    X_test = classifier.X_test
    y = classifier.y
    y_train = classifier.y_train
    y_val = classifier.y_val
    y_test = classifier.y_test
    
    #classifier.plot_random(no_of_spectra = 5)

    hist = classifier.train(checkpoint = True, early_stopping = False,
                            tb_log = True, csv_log = True)
    epochs_trained = len(hist['loss'])
    score = classifier.evaluate()
    test_loss, test_accuracy = score[0], score[1]

    pred_train, pred_test = classifier.predict()
    pred_train_classes, pred_test_classes = classifier.predict_classes()
    
    classifier.save_model()
    
    graphs = TrainingGraphs(classifier.history,
                            classifier.model_name,
                            classifier.time)
    
    classifier.shelve_results(full = False)
    
    dir_name = classifier.time + '_' + classifier.model_name
    rep = Report(dir_name)  
    rep.write()      
 

#%% Resume training with the same classifier, on the same model
# =============================================================================
# classifier.load_model(from_path = False)
# 
# #classifier2.load_model()
# hist = classifier.train(checkpoint = False,
#                         tb_log = True,
#                         early_stopping = True)
# score = classifier.evaluate()
# test_loss, test_accuracy = score[0], score[1]
# 
# pred_train, pred_test = classifier.predict()
# pred_train_classes, pred_test_classes = classifier.predict_classes()
# 
# graphs = TrainingGraphs(hist,classifier.model_name, classifier.time)
# classifier.save_model()
# =============================================================================

#%% Resume training with the same classifier, on the same model
# =============================================================================
# model_path = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\xpsdeeplearning\saved_models\20200608_17h51m_Fe_single_4_classes_CNN_simple' 
# classifier.load_model(from_path = True, model_path = model_path)
# 
# #classifier2.load_model()
# hist = classifier.train(checkpoint = False,
#                         tb_log = True,
#                         early_stopping = True)
# score = classifier.evaluate()
# test_loss, test_accuracy = score[0], score[1]
# 
# pred_train, pred_test = classifier.predict()
# pred_train_classes, pred_test_classes = classifier.predict_classes()
# 
# graphs = TrainingGraphs(hist,classifier.model_name, classifier.time)
# classifier.save_model()
# =============================================================================

#%% Resume training with a new classifier instance
# =============================================================================
# time =  '20200609_14h04m'
# model_type = 'CNN_simple'
# model_name = 'Fe_single_4_classes_CNN_simple'
#     
# classifier2 = Classifier(time = time,
#                         model_type = model_type,
#                         model_name = model_name)
# 
# hist2 = classifier2.train(checkpoint = False,
#                          tb_log = True,
#                          early_stopping = True)
# score2 = classifier2.evaluate()
# test_loss2, test_accuracy2 = score[0], score[1]
# 
# pred_train2, pred_test2 = classifier.predict()
# pred_train_classes2, pred_test_classes2 = classifier.predict_classes()
# 
# graphs2 = TrainingGraphs(hist2,classifier2.model_name, classifier2.time)
# classifier2.save_model()
# =============================================================================