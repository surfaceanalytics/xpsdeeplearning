# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:25 2020

@author: pielsticker
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

#%%
class EmptyModel(Model):
    def __init__(self, inputs, outputs, inputshape,
                 num_classes, no_of_inputs = 1, name = 'New_Model'):
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = no_of_inputs
                
        
        super(EmptyModel, self).__init__(inputs = inputs,
                                         outputs = outputs,
                                         name = name)
        
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super(EmptyModel, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        config['no_of_inputs'] = self.no_of_inputs
        
        return config

    
class CustomSimpleCNN(EmptyModel):
    def __init__(self, inputshape, num_classes):
        no_of_inputs = 1

        input_1 = Input(shape = inputshape)
        
        # Convolutional layers       
        conv_1 = Conv1D(32, 9, activation = 'relu')(input_1)
        conv_2 = Conv1D(32, 9, activation = 'relu')(conv_1)
        max_pool_1 = MaxPooling1D()(conv_2)
        drop_1 = Dropout(0.2)(max_pool_1)
        
        # Fully-connected layer
        flatten_1 = Flatten()(drop_1)
        dense_1 = Dense(128, activation = 'relu')(flatten_1)
        drop_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(num_classes, activation = 'softmax')(drop_2) 
        
        super(CustomSimpleCNN, self).__init__(inputs = input_1,
                                              outputs = dense_2,
                                              inputshape = inputshape,
                                              num_classes = num_classes,
                                              no_of_inputs = no_of_inputs, 
                                              name = 'Custom_CNN_Simple')
        

# =============================================================================
# class CustomCNN(EmptyModel):
#     def __init__(self, inputshape, num_classes):
#         no_of_inputs = 1
#         
#         input_1 = Input(shape = self.inputshape)
#          
#         # Convolutional layers
#         conv_1 = Conv1D(2, 9, activation = 'relu')(input_1)
#         average_pool_1 = AveragePooling1D()(conv_1)
#         batch_norm_1 = BatchNormalization()(average_pool_1)
#         
#         conv_2 = Conv1D(2, 7, activation = 'relu')(batch_norm_1)
#         average_pool_2 = AveragePooling1D()(conv_2)
#         batch_norm_2 = BatchNormalization()(average_pool_2)
#         
#         conv_3 = Conv1D(4, 7, activation = 'relu')(batch_norm_2)
#         average_pool_3 = AveragePooling1D()(conv_3)
#         batch_norm_3 = BatchNormalization()(average_pool_3)
#         
#         conv_4 = Conv1D(4, 5, activation = 'relu')(batch_norm_3)
#         average_pool_4 = AveragePooling1D()(conv_4)
#         batch_norm_4 = BatchNormalization()(average_pool_4)
#         
#         # Fully-connected layer
#         flatten_1 = Flatten()(batch_norm_4)
#         dense_1 = Dense(10, activation = 'relu')(flatten_1)
#         dense_2 = Dense(5, activation = 'relu')(dense_1) 
#         dense_3 = Dense(self.num_classes, activation = 'softmax')(dense_2)
#         
#         super(CustomCNN, self).__init__(inputs = input_1,
#                                         outputs = dense_3,
#                                         inputshape = inputshape,
#                                         num_classes = num_classes,
#                                         no_of_inputs = no_of_inputs, 
#                                         name = 'Custom_CNN')
# 
# =============================================================================

class CustomCNN(EmptyModel):
    def __init__(self, inputshape, num_classes):      
        input_1 = Input(shape = inputshape)
                
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
        dense_1 = Dense(1000, activation = 'relu')(drop_1)
        
        dense_2 = Dense(num_classes, activation = 'softmax')(dense_1)
        
        no_of_inputs = len(sublayers)

        super(CustomCNN, self).__init__(inputs = input_1,
                                        outputs = dense_2,
                                        inputshape = inputshape,
                                        num_classes = num_classes,
                                        no_of_inputs = no_of_inputs, 
                                        name = 'Custom_CNN')       
      
        
class CustomCNNMultiple(EmptyModel):
    def __init__(self, inputshape, num_classes):      
        input_1 = Input(shape = inputshape)
                
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
        
        dense_2 = Dense(num_classes, activation = 'softmax')(dense_1)
        
        norm_1 = LayerNormalization(trainable = False,
                                    name = 'output_norm')(dense_2)        
    
        no_of_inputs = len(sublayers)

        super(CustomCNNMultiple, self).__init__(inputs = input_1,
                                           outputs = norm_1,
                                           inputshape = inputshape,
                                           num_classes = num_classes,
                                           no_of_inputs = no_of_inputs, 
                                           name = 'Custom_CNN_multiple')       
            
        
    
class CustomMLP(EmptyModel):
    def __init__(self, inputshape, num_classes):
        no_of_inputs = 1
        
        input_1 = Input(shape = inputshape)
        
        flatten_1 = Flatten()(input_1)
        drop_1 = Dropout(0.5)(flatten_1)
        dense_1 = Dense(64, activation = 'relu')(drop_1)
        batch_norm_1 = BatchNormalization()(dense_1)
        dense_2 = Dense(num_classes, activation = 'softmax')(batch_norm_1)
        
        super(CustomMLP, self).__init__(inputs = input_1,
                                        outputs = dense_2,
                                        inputshape = inputshape,
                                        num_classes = num_classes,
                                        no_of_inputs = no_of_inputs, 
                                        name = 'Custom_MLP')     
      
        
#%% 
if __name__ == "__main__":
    input_shape = (1121,1)
    num_classes = 4
    model = CustomCNNMultiple(input_shape,num_classes)
    model.summary()
    
# =============================================================================
#     import os
#     filepath = os.getcwd()
#     model.save(filepath)
#     inputs = {'model': model.no_of_inputs}
#     inputs['model_config'] = model.get_config()['no_of_inputs']
# 
#     custom_objects = {'EmptyModel' : EmptyModel}
#     custom_objects[str(type(model).__name__)] =\
#         type(model).__name__
#     print(custom_objects)
#     
#     from tensorflow.keras.models import load_model
# 
#     loaded_model = load_model(filepath, custom_objects = custom_objects)
#     loaded_model.summary()
#     #inputs['loaded_model'] = loaded_model.no_of_inputs
#     inputs['loaded_model_config'] = loaded_model.get_config()['no_of_inputs']
#     
#     
#     no_of_drop_layers = 2
#     new_model = EmptyModel(
#         inputs = loaded_model.input,
#         outputs = loaded_model.layers[-no_of_drop_layers].output,
#         inputshape = input_shape,
#         num_classes = num_classes,
#         no_of_inputs = loaded_model.get_config()['no_of_inputs'],
#         name = 'Changed_Model')
#     new_model.summary()
#     inputs['new_model'] = new_model.no_of_inputs
#     inputs['new_model_config'] = new_model.get_config()['no_of_inputs']
#     
# 
# =============================================================================
