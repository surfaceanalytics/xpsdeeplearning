# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:25 2020

@author: pielsticker
"""

import os 

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

 
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

# Run tensorflow on local CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D as Convolution1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

       
class CustomModel(Sequential):
    def __init__(self, name = None):
        super(CustomModel, self).__init__(name = name)
        self.opt = Adam(learning_rate = 0.00001)
        
    def name_layers(self):
        for i, layer in enumerate(self.layers):
            layer.name = 'Layer' + str(i)
        
        # Name activation layer
        self.layers[-1].name += ': Activation'
        
    def print_shapes(self):
        for i, layer in enumerate(self.layers):
            print('Layer' + str(i) + ': ' + str(layer.input_shape))            
            print('Layer' + str(i) + ': ' + str(layer.output_shape))
            
                    

class CustomModelSimpleCNN(CustomModel):
    def __init__(self, inputshape, num_classes, name):
        super(CustomModelSimpleCNN, self).__init__(name = name)
        self.inputshape = inputshape
        self.num_classes = num_classes
         
        self.add(Convolution1D(32, 9,
                              activation = 'relu',
                              input_shape = self.inputshape))
        self.add(Convolution1D(64, 9, activation='relu'))
        self.add(MaxPooling1D())
        self.add(Dropout(0.25))
        self.add(Flatten())
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax')) 
        
        #self.name_layers()
        # Define adam parameters

        self.compile(loss = 'categorical_crossentropy',
                     optimizer = self.opt, 
                     metrics = ['accuracy'])
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomModelSimpleCNN, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        
        return config
        
    
        
class CustomModelCNN(CustomModel):
    def __init__(self, input_shape, num_classes, name):
        super(CustomModelCNN, self).__init__(name = name)
        
        # Convolutional layers - feature extraction
        self.add(Convolution1D(2, 9, input_shape = input_shape))   
        self.add(AveragePooling1D())
        self.add(BatchNormalization())

        self.add(Convolution1D(2, 7, activation='relu'))
        self.add(AveragePooling1D())
        self.add(BatchNormalization())

        self.add(Convolution1D(4, 7, activation='relu'))
        self.add(AveragePooling1D())
        self.add(BatchNormalization())

        self.add(Convolution1D(4, 5, activation='relu'))
        self.add(MaxPooling1D())
        self.add(BatchNormalization())

        # Fully-connected layer
        self.add(Flatten())
        self.add(Dense(10, activation='relu'))
        self.add(Dense(5, activation='relu'))

        # Output layer with softmax activation
        self.add(Dense(num_classes, activation='softmax'))
        
        
        self.name_layers()        
        
        self.compile(loss = 'categorical_crossentropy',
                     optimizer = self.opt, 
                     metrics = ['accuracy'])
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['input_shape'] = self.input_shape
        config['num_classes'] = self.num_classes
        
        return config



class CustomModelMLP(CustomModel):
    def __init__(self, input_shape, num_classes, name):
        super(CustomModelMLP, self).__init__(name = name)

        self.add(Flatten(input_shape = input_shape))

        self.add(Dropout(0.5))
        self.add(Dense(64, activation='relu'))
        self.add(BatchNormalization())

        self.add(Dropout(0.5))
        self.add(Dense(64, activation='relu'))
        self.add(BatchNormalization())
                    
        # Output layer
        self.add(Dense(num_classes, activation='softmax'))
        
        self.name_layers()
   
        self.compile(loss = 'categorical_crossentropy',
                     optimizer = self.opt, 
                     metrics = ['accuracy'])
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['input_shape'] = self.input_shape
        config['num_classes'] = self.num_classes
        
        return config
