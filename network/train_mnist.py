# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:16:51 2020

@author: pielsticker
"""


import os
import numpy as np
import datetime
from matplotlib import pyplot as plt

from sklearn.utils import shuffle

import tensorflow as tf
# Run tensorflow on local CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#from tensorflow import keras
from keras.models import Sequential
# from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K



#%%

class Classifier():
    def __init__(self, model_type ='CNN', model_name = 'Classifier'):
        self.model_name = model_name
        self.time = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm")
        
        self.epochs = 20
        self.batch_size = 16
        self.train_val_split_percentage = 0.2
        self.num_classes = 10
        
        # input image dimensions
        img_rows, img_cols = 28, 28
        

        # the data, split between train and test sets
        (X_train, y_train), (X_test, y_test) = \
            mnist.load_data()
        
        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows*img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows*img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train/= 255
        X_test /= 255
        
# =============================================================================
#         # Take only some classes
#         train_filter = np.where((y_train == 0 ) | (y_train == 1))
#         test_filter = np.where((y_test == 0) | (y_test == 1))
#         X_train, y_train = X_train[train_filter], y_train[train_filter]
#         X_test, y_test = X_test[test_filter], y_test[test_filter]
# =============================================================================

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        X_train, y_train = shuffle(X_train, y_train) 
        X_test, y_test = shuffle(X_test, y_test) 
        
        X_train = X_train[:1000, :, :]
        X_test = X_test[:4000, :, :]
        y_train = y_train[:1000, :]
        y_test = y_test[:4000, :]
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print('\n')
        print("Data was loaded!")
        print('No. of training samples: ' + str(len(self.X_train)))
        print('No. of test samples: ' + str(len(self.X_test)))
        print('Shape of each sample : '
              + str(self.X_train[0].shape[0]) + ' features (X)' 
              + ' + ' + str(self.y_train[0].shape[0])
              + ' labels (y)')
                    
        no_of_points = self.X_train.shape[1]
        
        input_shape = (no_of_points, 1)
        
        if model_type == 'CNN_simple':
            self.architecture = ArchitectureSimpleCNN(input_shape,
                                                      self.num_classes)
            self.model = self.architecture.model
            
        elif model_type == 'CNN':
            self.architecture = ArchitectureCNN(input_shape,
                                                self.num_classes)
            self.model = self.architecture.model
            
        elif model_type == 'MLP':
            self.architecture = ArchitectureMLP(input_shape,
                                                self.num_classes)
            self.model = self.architecture.model
            

        self.architecture.summary()
        self.architecture.save_and_print_architecture(self.model_name,
                                                      self.time)
        
        
    def check_class_distribution(self):
        class_distribution = {'training data': {},
                              'test data': {}}
        
        for i in range(self.num_classes):
            class_distribution['training data'][str(i)] = 0
            class_distribution['test data'][str(i)] = 0
             
        data_list = [self.y_train, self.y_test]
       
        for i, dataset in enumerate(data_list):
            key = list(class_distribution.keys())[i]
            
            for j in range(dataset.shape[0]): 
                argmax_class = np.argmax(dataset[j,:], axis = 0)
                class_distribution[key][str(argmax_class)] +=1
                
        return class_distribution
        

    def train(self, tb_log = False):
        callbacks = []
        
        if tb_log:
            logs_dir = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'network')[0] + 'logs\\'
            
            log_dir = logs_dir + self.time + '_' + self.model_name
            
            tensorboard_callback = TensorBoard(log_dir = log_dir,
                                               histogram_freq = 1,
                                               write_graph = True,
                                               write_images = True)
            callbacks.append(tensorboard_callback)
            
        # fit and run model  
        self.training = self.model.fit(self.X_train,
                                       self.y_train,
                                       validation_split = \
                                           self.train_val_split_percentage,
                                       nb_epoch = self.epochs,
                                       batch_size = self.batch_size,
                                       verbose = True,
                                       callbacks = callbacks)
        print("Training done!")
        
        return self.training.history
        
    
    def evaluate(self):
        loss = self.model.evaluate(self.X_test,
                                   self.y_test,
                                   batch_size = self.batch_size,
                                   verbose=True)
        
        return loss
     
     
    def predict(self):
        prediction_training = self.model.predict(self.X_train, verbose = True)
        prediction_test = self.model.predict(self.X_test, verbose = True)
        
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
        
        return pred_training_classes, pred_test_classes
            
    
    def save_model(self):
        model_dir = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'network')[0] + 'saved_models\\'
        
        t = self.time
        file_name = model_dir + t + '_' + self.model_name + '_model.json'
        weights_name = model_dir + t + '_' + self.model_name + "_weights.h5"
        param_name = model_dir + t + self.model_name + '_hyperparams.json'
        
        params_dict = {
            'train_val_split_percentage' : self.train_val_split_percentage,
            'epochs_trained' :self.epochs,
            'batch_size' : self.batch_size,
            'optimizer' : {'name' : 'adam',
                           'learning_rate' : 0.001,
                           'beta_1' : 0.9,
                           'beta_2' : 0.999,
                           'amsgrad' : False}}
             
        model_json = self.model.to_json()
        with open(file_name, 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_name)
        print("Saved model and hyperparameters.")
            
            
    def load_model(self):
        model_dir = os.path.dirname(
                os.path.abspath(__file__)).partition(
                        'network')[0] + 'saved_models\\'
        t = self.time
        
        file_name = model_dir + t + '_' + self.model_name + '_model.json'
        weights_name = model_dir + t + '_' + self.model_name + "_weights.h5"
        
        # load json and create model
        with open(file_name, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_name)
        
        self.model = loaded_model
        print("Loaded model from disk.")
        
        
    def plot_random(self,no_of_spectra):        
        for i in range(no_of_spectra):
            r = np.random.randint(0,X_train.shape[0])
            y = self.X_train[r]
            label = str(self.y_train[r])
            fig = SpectrumFigure(y, label)
            plt.show()
        
        
    
class Architecture():
    def __init__(self, input_shape, num_classes):
        self.model = Sequential() 
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def summary(self):
        print(self.model.summary())
        print("Model created!")
    
    def save_and_print_architecture(self, model_name, time):
        fig_folder = os.path.dirname(
            os.path.abspath(__file__)).partition(
                'xpsdeeplearning')[0] + 'figures\\'
        fig_file_path = fig_folder + time + '_' + model_name  
        os.makedirs(fig_file_path)
        fig_file_name = fig_file_path+ '\\model.png'
        plot_model(self.model, to_file = fig_file_name,
                   rankdir = "LR", show_shapes = True)
        model_plot = plt.imread(fig_file_name)
        plt.imshow(model_plot)
        plt.show()
        
    def name_layers(self):
        for i, layer in enumerate(self.model.layers):
            layer.name = 'Layer' + str(i)
        
        # Name activation layer
        self.model.layers[-1].name += ': Activation'
        
    def print_shapes(self):
        for i, layer in enumerate(self.model.layers):
            print('Layer' + str(i) + ': ' + str(layer.input_shape))            
            print('Layer' + str(i) + ': ' + str(layer.output_shape))
        

        
class ArchitectureSimpleCNN(Architecture):
     def __init__(self, input_shape, num_classes):
        Architecture.__init__(self, input_shape, num_classes)
        self.model.add(Convolution1D(32, 9,
                              activation='relu',
                              input_shape=input_shape))
        self.model.add(Convolution1D(64, 9, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax')) 
        
        self.name_layers()
        # Define adam parameters
        self.adam_opt = Adam(learning_rate = 0.0001,
                             beta_1 = 0.9,
                             beta_2 = 0.999,
                             amsgrad = False)
        
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = self.adam_opt, 
                           metrics = ['accuracy'])
    
        
class ArchitectureCNN(Architecture):
    def __init__(self, input_shape, num_classes):
        Architecture.__init__(self, input_shape, num_classes)
        
        # Convolutional layers - feature extraction
        self.model.add(Convolution1D(2, 9,
                                     input_shape = self.input_shape))   
        self.model.add(AveragePooling1D())
        self.model.add(BatchNormalization())

        self.model.add(Convolution1D(2, 7, activation='relu'))
        self.model.add(AveragePooling1D())
        self.model.add(BatchNormalization())

        self.model.add(Convolution1D(4, 7, activation='relu'))
        self.model.add(AveragePooling1D())
        self.model.add(BatchNormalization())

        self.model.add(Convolution1D(4, 5, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(BatchNormalization())

        # Fully-connected layer
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(5, activation='relu'))

        # Output layer with softmax activation
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        
        self.name_layers()        

        # Define adam parameters
        self.adam_opt = Adam(learning_rate = 0.001,
                             beta_1 = 0.9,
                             beta_2 = 0.999,
                             amsgrad = False)
        
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = self.adam_opt, 
                           metrics = ['accuracy'])


class ArchitectureMLP(Architecture):
    def __init__(self, input_shape, num_classes):
        Architecture.__init__(self, input_shape, num_classes)

        self.model.add(Flatten(input_shape=self.input_shape))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
                    
        # Output layer
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        self.name_layers()

        # Define adam parameters
        self.adam_opt = Adam(learning_rate = 0.001,
                             beta_1 = 0.9,
                             beta_2 = 0.999,
                             amsgrad = False)
        
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = self.adam_opt, 
                           metrics = ['accuracy'])
    

class TrainingGraphs():
    def __init__(self, history, model_name, time):
        self.history = history
       
        fig_folder = os.path.dirname(
            os.path.abspath(__file__)).partition(
                'network')[0] + 'figures\\'
        self.fig_file_name = fig_folder + time + '_' + model_name 
        os.makedirs(self.fig_file_name)
        
        self.plot_loss()
        self.plot_accuracy()
        
    def plot_loss(self):
        # summarize history for loss
        plt.figure(figsize=(9, 5))
        plt.plot(self.history['loss'], linewidth = 3)
        plt.plot(self.history['val_loss'], linewidth = 3)
        plt.title('Loss')
        plt.ylabel('Cross Entropy Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        fig_name = self.fig_file_name + '\\loss.png' 
        plt.savefig(fig_name)
        plt.show()

    def plot_accuracy(self):
        #summarize history for accuracy
        plt.figure(figsize=(9, 5))
        plt.plot(self.history['accuracy'], linewidth = 3)
        plt.plot(self.history['val_accuracy'], linewidth = 3)
        plt.title('Accuracy')
        plt.ylabel('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        fig_name = self.fig_file_name + '\\accuracy.png' 
        plt.savefig(fig_name)
        plt.show()
        
        
        
        
class SpectrumFigure:
    def __init__(self, y, fig_text):
        self.x = np.arange(694,750.05,0.05)
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=(5,4), dpi=100)
        self.fig.patch.set_facecolor('0.9411')
        self.ax.plot(np.flip(self.x),self.y)
        self.ax.invert_xaxis()
        self.ax.set_xlabel('Binding energy (eV)')
        self.ax.set_ylabel('Intensity (arb. units)')
        self.fig.tight_layout()
        self.fig_text = fig_text
        self.ax.text(0.1, 0.45,fig_text,
                     horizontalalignment='left',
                     verticalalignment='top',
                     transform = self.ax.transAxes,
                     fontsize = 9)


#%% 
if __name__ == "__main__":
    np.random.seed(100)
    classifier = Classifier(model_type = 'CNN_simple',
                            model_name = 'MNIST_CNN_simple')
    
    class_distribution = classifier.check_class_distribution()

    X_train = classifier.X_train
    X_test = classifier.X_test
    y_train = classifier.y_train
    y_test = classifier.y_test
                   
    hist = classifier.train(tb_log = True)
    score = classifier.evaluate()
    test_loss, test_accuracy = score[0], score[1]
    pred_train, pred_test = classifier.predict()
    pred_train_classes = classifier.model.predict_classes(X_train)
    pred_test_classes = classifier.model.predict_classes(X_test)
    
    classifier.save_model()
    graphs = TrainingGraphs(hist,classifier.model_name, classifier.time)
