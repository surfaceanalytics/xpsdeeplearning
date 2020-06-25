# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:25 2020

@author: pielsticker
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization

#%%
class CustomSequential(Sequential):
    def __init__(self, inputshape, num_classes, name = None):
        super(CustomSequential, self).__init__(name = name)
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = 1
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomSequential, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        
        return config
      
    
class CustomSimpleCNN(CustomSequential):
    def __init__(self, inputshape, num_classes):
        super(CustomSimpleCNN, self).__init__(inputshape,
                                               num_classes,
                                               name = 'Custom_CNN_Simple')
        self.add(Conv1D(32, 9,
                        activation = 'relu',
                        input_shape = self.inputshape))
        self.add(Conv1D(64, 9, activation='relu'))
        self.add(MaxPooling1D())
        self.add(Dropout(0.25))
        self.add(Flatten())
        self.add(Dense(128, activation = 'relu'))
        self.add(Dropout(0.5))
        self.add(Dense(self.num_classes, activation = 'softmax')) 
        
    

class CustomCNN(CustomSequential):
    def __init__(self, inputshape, num_classes):
        super(CustomCNN, self).__init__(inputshape,
                                        num_classes,
                                        name = 'Custom_CNN')
        # Convolutional layers - feature extraction
        self.add(Conv1D(2, 9,
                        activation = 'relu',
                        input_shape = self.inputshape))   
        self.add(AveragePooling1D())
        self.add(BatchNormalization())
 
        self.add(Conv1D(2, 7, activation = 'relu'))
        self.add(AveragePooling1D())
        self.add(BatchNormalization())

        self.add(Conv1D(4, 7, activation = 'relu'))
        self.add(AveragePooling1D())
        self.add(BatchNormalization())
 
        self.add(Conv1D(4, 5, activation = 'relu'))
        self.add(MaxPooling1D())
        self.add(BatchNormalization())

        # Fully-connected layer
        self.add(Flatten())
        self.add(Dense(10, activation = 'relu'))
        self.add(Dense(5, activation = 'relu'))

        # Output layer with softmax activation
        self.add(Dense(self.num_classes, activation = 'softmax'))



class CustomCNNSub(Model):
    def __init__(self, inputshape, num_classes, name = None):
        self.inputshape = inputshape
        self.num_classes = num_classes
        
        input_layer = Input(shape = self.inputshape)
                
        conv_short = Conv1D(4, 5, padding = 'same',
                            activation = 'relu')(input_layer)
        conv_medium = Conv1D(4, 10, padding = 'same',
                             activation = 'relu')(input_layer)
        conv_long = Conv1D(4, 15, padding = 'same',
                           activation = 'relu')(input_layer)

        sublayers = [conv_short, conv_medium, conv_long]
        merged_sublayers = concatenate(sublayers)
        
        conv_all = Conv1D(4, 5, activation='relu')(merged_sublayers)
        pool_all = AveragePooling1D()(conv_all)
        flatten = Flatten()(pool_all)
        drop = Dropout(0.2)(flatten)
        first_dense = Dense(2000, activation = 'relu')(drop)
        output = Dense(self.num_classes, activation = 'softmax')(first_dense)

        super(CustomCNNSub, self).__init__(
            inputs = input_layer,
            outputs = output,
            name = 'Custom_CNN_Sub')
        
        self.no_of_inputs = len(sublayers)
            
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(CustomCNNSub, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        
        return config


        
class CustomMLP(CustomSequential):
    def __init__(self, inputshape, num_classes):
        super(CustomMLP, self).__init__(inputshape,
                                        num_classes,
                                        name = 'Custom_MLP')
        self.add(Flatten(input_shape = self.inputshape))

        self.add(Dropout(0.5))
        self.add(Dense(64, activation = 'relu'))
        self.add(BatchNormalization())
       
        self.add(Dropout(0.5))
        self.add(Dense(64, activation = 'relu'))
        self.add(BatchNormalization())
       
        self.add(Dense(self.num_classes, activation = 'softmax'))
        
        #self.name_layers()
      
        
#%% 
if __name__ == "__main__":
    input_shape = (1121,1)
    num_classes = 4
    model = CustomMLP(input_shape,num_classes)
    model.summary()