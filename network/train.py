# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:16:51 2020

@author: pielsticker
"""


import os
import sys
import numpy as np
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from matplotlib import pyplot as plt
# from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Dense, Input
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D
from keras.utils import plot_model
from keras.optimizers import Adam

#%%

class ClassifierCNN():
    def __init__(self):
        self.epochs = 1000
        self.batch_size = 50
        self.train_test_split_percentage = 0.2
        
        #load dataset and preprocess it
        #self.X_train, self.X_test, self.y_train, self.y_test = self.load_data_preprocess(
        #    self.train_test_split_percentage)
        
        #no_of_points = self.X_train.shape[1]
        no_of_points = 1200
        self.cnn_model = ArchitectureCNN(no_of_points)
        self.cnn_model.summary()
        self.cnn_model.print_architecture()
        
    def train(self):        
        # fit and run model    
        training = self.cnn_model.fit(self.X_train,
                                      self.y_train,
                                      validation_data = (self.X_test, self.y_test),
                                      nb_epoch = self.epochs,
                                      batch_size = self.batch_size,
                                      verbose=True)
        
        print("Training done!")
        graphs = TrainingGraphs(training.history)
        graphs.plot_training_graphs()
        
    def load_data_preprocess(self):
        pass



class ArchitectureCNN():
    def __init__(self, no_of_points):
        self.model = Sequential() 
        

        # Definition of the input_shape
        input_shape=(no_of_points,1)
        

# =============================================================================
#         self.model.add(Dense(shape=input_shape))
#         self.model.add(Dense(1,input_shape=(no_of_points,1), activation=activation))
# =============================================================================
        
        # Convolutional layers - feature extraction
        self.model.add(Convolution1D(2, 9,
                                     input_shape = input_shape,
                                     activation='relu'))
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
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(5, activation='relu'))


        self.model.add(Dropout(0.10))
        self.model.add(Convolution1D(3, 1))
        # model.add(GlobalAveragePooling1D())

       # Output layer with softmax activation
        self.model.add(Dense(4, activation='softmax', name='Activation'))
        
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
        self.model.compile(
            optimizer= adam_opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def summary(self):
        print(self.model.summary())
        print("CNN Model created.")
        
    def print_architecture(self):
        plot_model(self.model, to_file='CNN_model.png')
        model_plot = plt.imread('CNN_model.png')
        plt.imshow(model_plot)
        plt.show()
     
        
class TrainingGraphs():
    def __init__(self, history):
        fig, ax = plt.figure(figsize=(15, 5))
    
    def plot_accuracy(hist):
    #summarize history for accuracy
        plt.subplot(1, 2, 1)
        plt.plot(hist.history['acc'], linewidth = 3)
        plt.title('Model Training Accuracy')
        plt.ylabel('Training Accuracy')
        plt.xlabel('Epoch')
        
    def plot_loss(hist):
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(hist.history['loss'], linewidth = 3)
        plt.title('Model Training Loss')
        plt.ylabel('Cross Entropy Loss')
        plt.xlabel('Epoch')
        plt.savefig('figures/training_accuracy.png')
        plt.show()

        plt.figure(figsize=(10, 8))

        plt.plot(hist.history['val_acc'], linewidth = 3)
        plt.plot(hist.history['acc'], linewidth = 3)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Test', 'Train'], loc='lower right')
        plt.savefig('figures/test_accuracy.png')
        plt.show()


if __name__ == "__main__":
    #Mn2_C = pd.read_pickle(os.path.join(path_to_input, 'Mn2_Larger_Clean_Thin.pkl'))
    classifier = ClassifierCNN()
    #classifier.train()
