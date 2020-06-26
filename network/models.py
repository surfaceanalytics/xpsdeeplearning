# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:25 2020

@author: pielsticker
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization

#%%    
class CustomSimpleCNN(Model):
    def __init__(self, inputshape, num_classes):
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = 1
        
        input_1 = Input(shape = self.inputshape)
        
        # Convolutional layers       
        conv_1 = Conv1D(32, 9, activation = 'relu')(input_1)
        conv_2 = Conv1D(32, 9, activation = 'relu')(conv_1)
        max_pool_1 = MaxPooling1D()(conv_2)
        drop_1 = Dropout(0.2)(max_pool_1)
        
        # Fully-connected layer
        flatten_1 = Flatten()(drop_1)
        dense_1 = Dense(128, activation = 'relu')(flatten_1)
        drop_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(self.num_classes, activation = 'softmax')(drop_2) 
        
        super(CustomSimpleCNN, self).__init__(inputs = input_1,
                                              outputs = dense_2,
                                              name = 'Custom_CNN_Simple')
        
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomSimpleCNN, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        config['no_of_inputs'] = self.no_of_inputs
        
        return config



class CustomCNN(Model):
    def __init__(self, inputshape, num_classes):
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = 1
        
        input_1 = Input(shape = self.inputshape)
         
        # Convolutional layers
        conv_1 = Conv1D(2, 9, activation = 'relu')(input_1)
        average_pool_1 = AveragePooling1D()(conv_1)
        batch_norm_1 = BatchNormalization()(average_pool_1)
        
        conv_2 = Conv1D(2, 7, activation = 'relu')(batch_norm_1)
        average_pool_2 = AveragePooling1D()(conv_2)
        batch_norm_2 = BatchNormalization()(average_pool_2)
        
        conv_3 = Conv1D(4, 7, activation = 'relu')(batch_norm_2)
        average_pool_3 = AveragePooling1D()(conv_3)
        batch_norm_3 = BatchNormalization()(average_pool_3)
        
        conv_4 = Conv1D(4, 5, activation = 'relu')(batch_norm_3)
        average_pool_4 = AveragePooling1D()(conv_4)
        batch_norm_4 = BatchNormalization()(average_pool_4)
        
        # Fully-connected layer
        flatten_1 = Flatten()(batch_norm_4)
        dense_1 = Dense(10, activation = 'relu')(flatten_1)
        dense_2 = Dense(5, activation = 'relu')(dense_1) 
        dense_3 = Dense(self.num_classes, activation = 'softmax')(dense_2)
        
        super(CustomCNN, self).__init__(inputs = input_1,
                                        outputs = dense_3,
                                        name = 'Custom_CNN')
        
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomSimpleCNN, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        config['no_of_inputs'] = self.no_of_inputs
        
        return config


class CustomCNNSub(Model):
    def __init__(self, inputshape, num_classes, name = None):      
        self.inputshape = inputshape
        self.num_classes = num_classes

        input_1 = Input(shape = self.inputshape)
                
        conv_1_short = Conv1D(4, 5, padding = 'same',
                            activation = 'relu')(input_1)
        conv_1_medium = Conv1D(4, 10, padding = 'same',
                             activation = 'relu')(input_1)
        conv_1_long = Conv1D(4, 15, padding = 'same',
                           activation = 'relu')(input_1)
        sublayers = [conv_1_short, conv_1_medium, conv_1_long]
        merged_sublayers = concatenate(sublayers)
        
        conv_2 = Conv1D(4, 5, activation='relu')(merged_sublayers)
        average_pool_1 = AveragePooling1D()(conv_2)
        
        flatten_1 = Flatten()(average_pool_1)
        drop_1 = Dropout(0.2)(flatten_1)
        dense_1 = Dense(2000, activation = 'relu')(drop_1)
        
        dense_2 = Dense(self.num_classes, activation = 'softmax')(dense_1)

        super(CustomCNNSub, self).__init__(
            inputs = input_1,
            outputs = dense_2,
            name = 'Custom_CNN_Sub')
        
        self.no_of_inputs = len(sublayers)
            
        
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomCNNSub, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        config['no_of_inputs'] = self.no_of_inputs

        return config

   
class CustomMLP(Model):
    def __init__(self, inputshape, num_classes):
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = 1
        
        input_1 = Input(shape = self.inputshape)
        
        flatten_1 = Flatten()(input_1)
        drop_1 = Dropout(0.5)(flatten_1)
        dense_1 = Dense(64, activation = 'relu')(drop_1)
        batch_norm_1 = BatchNormalization()(dense_1)
        dense_2 = Dense(self.num_classes, activation = 'softmax')(batch_norm_1)

        super(CustomMLP, self).__init__(inputs = input_1,
                                        outputs = dense_2,
                                        name = 'Custom_MLP')
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomCNNSub, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        config['no_of_inputs'] = self.no_of_inputs

        return config
      
        
#%% 
if __name__ == "__main__":
    input_shape = (1121,1)
    num_classes = 4
    model = CustomCNNSub(input_shape,num_classes)
    model.summary()