# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:16:51 2020

@author: pielsticker
"""


import os
import sys
import numpy as np
import pandas as pd
import json
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from matplotlib import pyplot as plt
# from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Dense, Input, Flatten
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D
from keras.utils import plot_model
from keras.optimizers import Adam

#%%

class ClassifierCNN():
    def __init__(self):
        self.epochs = 50
        self.batch_size = 50
        self.train_test_split_percentage = 0.2
        self.label_values = ['Fe metal','FeO','Fe3O4','Fe2O3']

        
        # load dataset and preprocess it
        #self.load_data_preprocess()        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.load_data_preprocess()
                    
        no_of_points = len(self.X_train[0])
        self.architecture = ArchitectureCNN(no_of_points)
        self.model = self.architecture.cnn_model
        
        self.architecture.summary()
        self.architecture.print_architecture()
        
    def train(self):        
        # fit and run model    
        self.training = self.model.fit(self.X_train,
                                       self.y_train,
                                       validation_data = (
                                           self.X_test, self.y_test),
                                       nb_epoch = self.epochs,
                                       batch_size = self.batch_size,
                                       verbose=True)
        print("Training done!")
        graphs = TrainingGraphs(self.training.history)
        graphs.plot_accuracy()
        graphs.plot_loss()
        
        return self.training
        
    def evaluate(self):
        loss = self.model.evaluate(self.X_test,
                                   self.y_test,
                                   batch_size = self.batch_size,
                                   verbose=True)
        
        return loss
        
    def load_data_preprocess(self):
        #input_datafolder = r'C:\Users\pielsticker\Lukas\MPI-CEC\Projects\DataScience\machinelearning\xpsdeeplearning\data\simulated\\'
        input_datafolder = r'C:\Users\pielsticker\simulations'
        filename_basic = input_datafolder + '\\' + '20200519_iron_single_'
             
        # Determine no. of simulations files
        #no_of_files = len(next(os.walk(input_datafolder))[2])
        no_of_files = 1
        

        self.X = []
        self.y = []  
        
        for i in range(no_of_files): 
            filename_load = filename_basic + str(i) + '.json'
            with open(filename_load) as json_file:
                test = json.load(json_file)
            for j in range(0,len(test)):
                X = test[j]['y']
                y = test[j]['label']
                self.X.append(X)
                self.y.append(y)
                print('Load: ' + str(i*len(test)+j+1) + '/' + \
                      str(no_of_files*len(test)))
                    
                    
        self.y = self._one_hot_encode(self.y, self.label_values)
        X_train, X_test, y_train, y_test = self._split_test_train(self.X,
                                                                  self.y)
        
        self.X = np.array(self.X)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
        self.y = np.array(self.y)


    

        print('\n')
        print("Data was loaded!")
        print('Total no. of examples: ' + str(len(self.X)))
        print('No. of training examples: ' + str(len(X_train)))
        print('No. of test examples: ' + str(len(X_test)))
        print('Shape of each example : '
              + str(X_train[0].shape[0]) + ' features (X)' 
              + ' + ' + str(y_train[0].shape[0])
              + ' labels (y)')

        
        return X_train, X_test, y_train, y_test
        
    
    def _one_hot_encode(self,y, label_values):
        """
        

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        new_labels : TYPE
            DESCRIPTION.

        """
        labels = [i for s in [d.keys() for d in y] for i in s]
        new_labels = []
        
        for label in labels:  
            number = [i for i,x in enumerate(label_values) if x == label][0]
            label_list = [0,0,0,0]
            label_list[number] = 1
            new_labels.append(label_list)
                        
        return new_labels
    
    
    def _split_test_train(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        X_train : TYPE
            DESCRIPTION.
        X_test : TYPE
            DESCRIPTION.
        y_train : TYPE
            DESCRIPTION.
        y_test : TYPE
            DESCRIPTION.

        """
        random.seed(1)
        X_rand = random.sample(X, len(X))
        y_rand = random.sample(y, len(y))
        no_of_train = int((1-self.train_test_split_percentage)*len(X))
        
        X_train = np.array(X_rand[:no_of_train])
        X_test = np.array(X_rand[no_of_train:])
        
        #X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        
        y_train = np.array(y_rand[:no_of_train])
        y_test = np.array(y_rand[no_of_train:])
                
        return X_train, X_test, y_train, y_test
    

class ArchitectureCNN():
    def __init__(self, no_of_points):
        self.cnn_model = Sequential() 
        

        # Definition of the input_shape
        input_shape= (no_of_points,1)
                
        # Convolutional layers - feature extraction
        self.cnn_model.add(Convolution1D(2, 9,
                                     input_shape = input_shape,
                                     activation='relu'))
        self.cnn_model.add(AveragePooling1D())
        self.cnn_model.add(BatchNormalization())

        self.cnn_model.add(Convolution1D(2, 7, activation='relu'))
        self.cnn_model.add(AveragePooling1D())
        self.cnn_model.add(BatchNormalization())

        self.cnn_model.add(Convolution1D(4, 7, activation='relu'))
        self.cnn_model.add(AveragePooling1D())
        self.cnn_model.add(BatchNormalization())

        self.cnn_model.add(Convolution1D(4, 5, activation='relu'))
        self.cnn_model.add(MaxPooling1D())
        self.cnn_model.add(BatchNormalization())

        # Fully-connected layer
        self.cnn_model.add(Dense(10, activation='relu'))
        self.cnn_model.add(Dense(5, activation='relu'))


        self.cnn_model.add(Dropout(0.10))
        self.cnn_model.add(Convolution1D(3, 1))
        # model.add(GlobalAveragePooling1D())

        # Output layer with softmax activation
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(4, activation='softmax', name='Activation'))
        
        # Define adam parameters
        adam_opt = Adam(learning_rate=0.001,
                        beta_1=0.9,
                        beta_2=0.999,                     
                        amsgrad=False)
        
# =============================================================================
#  The layers were initialised from a
# Gaussian distribution with a zero mean and variance equal to
# 0.05.
# =============================================================================
        self.cnn_model.compile(
            optimizer= adam_opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def summary (self):
        print(self.cnn_model.summary())
        print("CNN Model created!")
        
    def print_architecture(self):
        plot_model(self.cnn_model, to_file='CNN_model.png')
        model_plot = plt.imread('CNN_model.png')
        plt.imshow(model_plot)
        plt.show()
     
        
class TrainingGraphs():
    def __init__(self, history):
        self.history = history
        plt.figure(figsize=(15, 5))
    
    def plot_accuracy(self):
        #summarize history for accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['acc'], linewidth = 3)
        plt.title('Model Training Accuracy')
        plt.ylabel('Training Accuracy')
        plt.xlabel('Epoch')
        
    def plot_loss(self):
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], linewidth = 3)
        plt.title('Model Training Loss')
        plt.ylabel('Cross Entropy Loss')
        plt.xlabel('Epoch')
        plt.savefig('figures/training_accuracy.png')
        plt.show()

        plt.figure(figsize=(10, 8))

        plt.plot(self.history['val_acc'], linewidth = 3)
        plt.plot(self.history['acc'], linewidth = 3)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Test', 'Train'], loc='lower right')
        plt.savefig('figures/test_accuracy.png')
        plt.show()


#%% 
if __name__ == "__main__":
    classifier = ClassifierCNN()
    training = classifier.train()
    #loss = classifier.evaluate()
