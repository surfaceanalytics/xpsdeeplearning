# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:25 2020

@author: pielsticker
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend as K

#%%
class EmptyModel(models.Model):
    """
    Base Model class to be used in the Classifier class of the 
    classifier module.
    """
    def __init__(self,
                 inputs,
                 outputs,
                 inputshape,
                 num_classes,
                 no_of_inputs=1,
                 name = 'New_Model'):
        """
        Aside from the inputs and outputs for the instantion of the 
        Model class from Keras, the EmptyModel class also gets as 
        paramters the input shape of the data, the no. of classes
        of the labels as well as how many times the input shall be 
        used.

        Parameters
        ----------
        inputs : keras.Input object or list of keras.Input objects.
            Inputs for the instantion of the Model class from Keras.
        outputs : Outputs of the last layer.
            Outputs for the instantion of the Model class from Keras.
        inputshape : ndarray
            Shape of the features of the training data set.
        num_classes : ndarray
            Shape of the labels of the training data set.
        no_of_inputs : int, optional
            Number of times the input shall be used in the Model.
            The default is 1.
        name : str, optional
            Name of the model.
            The default is 'New_Model'.

        Returns
        -------
        None.

        """
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = no_of_inputs
                
        
        super(EmptyModel, self).__init__(inputs=inputs,
                                         outputs=outputs,
                                         name=name)
        
    def get_config(self):
        """
        For serialization, all input paramters of the model are added to
        the get_config method from the keras.Model class.

        Returns
        -------
        config : dict
            Configuration of the model.

        """
        # For serialization with 'custom_objects'
        config = super(EmptyModel, self).get_config()
        config['inputshape'] = self.inputshape
        config['num_classes'] = self.num_classes
        config['no_of_inputs'] = self.no_of_inputs
        
        return config


class CustomMLP(EmptyModel):
    """
    A vanilla neural net with some hidden layers, but no convolutions.
    """
    def __init__(self, inputshape, num_classes):        
        self.input_1 = layers.Input(shape=inputshape,
                               name='input_1')
        
        self.flatten_1 = layers.Flatten(name='flatten1')(self.input_1)
        self.drop_1 = layers.Dropout(rate=0.5,
                                     name='drop_1')(self.flatten_1)
        self.dense_1 = layers.Dense(units=64,
                                    activation='relu',
                                    name='dense1')(self.drop_1)
        self.batch_norm_1 = layers.BatchNormalization(
            name='batch_norm_1')(self.dense_1)
        self.dense_2 = layers.Dense(units=num_classes,
                                    activation='softmax',
                                    name='dense2')(self.batch_norm_1)
        
        super(CustomMLP, self).__init__(inputs = self.input_1,
                                        outputs = self.dense_2,
                                        inputshape = inputshape,
                                        num_classes = num_classes, 
                                        name = 'Custom_MLP')      


class CustomCNN(EmptyModel):
    """
    A CNN with three convolutional layers of different kernel size at 
    the beginning. Works well for learning across scales.
    
    This is to be used for classification -> softmax activation in the
    last layer
    """
    def __init__(self, inputshape, num_classes):   
        self.input_1 = layers.Input(shape = inputshape)
        
        self.conv_1_short = layers.Conv1D(filters=4,
                                          kernel_size=5,
                                          strides=1,
                                          padding='same',
                                          activation='relu',
                                          name='conv_1_short')(self.input_1)
        self.conv_1_medium = layers.Conv1D(filters=4,
                                           kernel_size=10,
                                           strides=1,
                                           padding='same',
                                           activation='relu',
                                           name='conv_1_medium')(self.input_1)
        self.conv_1_long = layers.Conv1D(filters=4,
                                         kernel_size=10,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         name='conv_1_long')(self.input_1)
        
        sublayers = [self.conv_1_short, self.conv_1_medium, self.conv_1_long]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = layers.Conv1D(filters=10,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',
                                    activation='relu',
                                    name='conv_2')(merged_sublayers)
        self.conv_3 = layers.Conv1D(filters=10,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',
                                    activation='relu',
                                    name="conv_3")(self.conv_2)
        self.average_pool_1 = layers.AveragePooling1D(
            name='average_pool_1')(self.conv_3)
        
        self.flatten_1 = layers.Flatten(name='flatten1')(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2,
                                     name='drop_1')(self.flatten_1)
        self.dense_1 = layers.Dense(units=1000,
                                    activation='relu',
                                    name='dense_1')(self.drop_1)
        self.dense_2 = layers.Dense(units=num_classes,
                                    activation='softmax',
                                    name='dense_2')(self.dense_1)
        
        no_of_inputs = len(sublayers)

        super(CustomCNN, self).__init__(inputs = self.input_1,
                                        outputs = self.dense_2,
                                        inputshape = inputshape,
                                        num_classes = num_classes,
                                        no_of_inputs = no_of_inputs, 
                                        name = 'Custom_CNN')  
      
        
class CustomCNNMultiple(EmptyModel):
    """
    A CNN with three convolutional layers of different kernel size at 
    the beginning. Works well for learning across scales.
    
    This is to be used for regression on all labels. -> sigmoid 
    activation in the last layer.
    """
    def __init__(self, inputshape, num_classes):      
        self.input_1 = layers.Input(shape = inputshape)
        
        self.conv_1_short = layers.Conv1D(filters=12,
                                          kernel_size=5,
                                          strides=1,
                                          padding='same',
                                          activation='relu',
                                          name='conv_1_short')(self.input_1)
        self.conv_1_medium = layers.Conv1D(filters=12,
                                           kernel_size=10,
                                           strides=1,
                                           padding='same',
                                           activation='relu',
                                           name='conv_1_medium')(self.input_1)
        self.conv_1_long = layers.Conv1D(filters=12,
                                         kernel_size=10,
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         name='conv_1_long')(self.input_1)
        
        sublayers = [self.conv_1_short, self.conv_1_medium, self.conv_1_long]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = layers.Conv1D(filters=10,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',
                                    activation='relu',
                                    name='conv_2')(merged_sublayers)
        self.conv_3 = layers.Conv1D(filters=10,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',
                                    activation='relu',
                                    name="conv_3")(self.conv_2)
        self.average_pool_1 = layers.AveragePooling1D(
            name='average_pool_1')(self.conv_3)
        
        self.flatten_1 = layers.Flatten(name='flatten1')(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2,
                                     name='drop_1')(self.flatten_1)
        self.dense_1 = layers.Dense(units=4000,
                                    activation='relu',
                                    name='dense_1')(self.drop_1)
        self.dense_2 = layers.Dense(units=num_classes,
                                    activation='sigmoid',
                                    name='dense_2')(self.dense_1)
        
        self.output_norm = layers.Lambda(
            lambda x: x/tf.reshape(K.sum(x, axis=-1),(-1,1)),
            name = 'output_normalization')(self.dense_2)

        no_of_inputs = len(sublayers)

        super(CustomCNNMultiple, self).__init__(inputs=self.input_1,
                                                outputs=self.output_norm,
                                                inputshape=inputshape,
                                                num_classes=num_classes,
                                                no_of_inputs=no_of_inputs,
                                                name='Custom_CNN_multiple') 


### RESNET50 implementation ###
class IdentityBlock(models.Model):   
    def __init__(self,
                 filters,
                 kernel_size_2,
                 stage,
                 block,
                 strides=1):
        
        name = 'stage_' + str(stage) + str(block) + '_id_block'
        super(IdentityBlock, self).__init__(name=name)
        
        # Store filters
        filter1, filter2, filter3 = filters
         
        ### Main Path ###
        # Component 1
        self.conv_1 = layers.Conv1D(filters=filter1,
                                    kernel_size=1,
                                    strides=strides,
                                    padding='valid',
                                    kernel_initializer=glorot_uniform(
                                        seed=0))
        self.batch_1 = layers.BatchNormalization(axis=1)
        # Component 2
        self.conv_2 = layers.Conv1D(filters=filter2,
                                    kernel_size=kernel_size_2,
                                    strides=strides,
                                    padding='same',
                                    kernel_initializer=glorot_uniform(
                                        seed=0))

        self.batch_2 = layers.BatchNormalization(axis=1)
          
        # Component 3
        self.conv_3 = layers.Conv1D(filters=filter3,
                                    kernel_size=1,
                                    strides=strides,
                                    padding='valid',
                                    kernel_initializer=glorot_uniform(
                                        seed=0))
        self.batch_3 = layers.BatchNormalization(axis=1)
          
    def call(self, input_tensor, training=False):
        # Intermediate save of the input.
        x = input_tensor
        x_shortcut = input_tensor
        
# =============================================================================
#         ### Main Path ###
#         x = self.conv_1(x)
#         x = self.batch_1(x, training=training)
#         x = tf.nn.relu(x)
#         
#         x = self.conv_2(x)
#         x = self.batch_2(x, training=training)
#         x = tf.nn.relu(x)
#         
#         x = self.conv_2(x)
#         x = self.batch_2(x, training=training)
#         
#         # Final step: Add shortcut to main path.
#         x += x_shortcut
#         #x = layers.Add()([x, x_shortcut])
# =============================================================================
        return tf.nn.relu(x)  


class ConvBlock(models.Model):   
    def __init__(self,
                 filters,
                 kernel_size_2,
                 stage,
                 block,
                 strides=1):
        
        name = 'stage_' + str(stage) + str(block) + '_conv_block'
        super(ConvBlock, self).__init__(name=name)
        
        # Store filters
        filter1, filter2, filter3 = filters
         
        ### Main Path ###
        # Component 1
        self.conv_1 = layers.Conv1D(filters=filter1,
                                    kernel_size=1,
                                    strides=strides,
                                    padding='valid',
                                    kernel_initializer=glorot_uniform(
                                        seed=0))
        self.batch_1 = layers.BatchNormalization(axis=1)          
        # Component 2
        self.conv_2 = layers.Conv1D(filters=filter2,
                                    kernel_size=kernel_size_2,
                                    strides=strides,
                                    padding='same',
                                    kernel_initializer=glorot_uniform(
                                        seed=0))

        self.batch_2 = layers.BatchNormalization(axis=1)
          
        # Component 3
        self.conv_3 = layers.Conv1D(filters=filter3,
                                    kernel_size=1,
                                    strides=strides,
                                    padding='valid',
                                    kernel_initializer=glorot_uniform(seed=0))
        self.batch_3 = layers.BatchNormalization(axis=1)
        
        ### Shortcut Path ###
        self.conv_short = layers.Conv1D(filters=filter3,
                                        kernel_size=1,
                                        strides=strides,
                                        padding='valid',
                                        kernel_initializer=glorot_uniform(
                                            seed=0))
        self.batch_short = layers.BatchNormalization(axis=1)                            
            
          
    def call(self, input_tensor, training=False):
        # Intermediate save of the input.
        x = input_tensor
        x_shortcut = input_tensor
        
# =============================================================================
#         ### Main Path ###
#         x = self.conv_1(x)
#         x = self.batch_1(x, training=training)
#         x = tf.nn.relu(x)
#         
#         x = self.conv_2(x)
#         x = self.batch_2(x, training=training)
#         x = tf.nn.relu(x)
#         
#         x = self.conv_2(x)
#         x = self.batch_2(x, training=training)
#         
#         ### Shortcut Path ### 
#         x_shortcut = self.conv_short(x_shortcut)
#         x_shortcut = self.batch_short(x_shortcut)
#         # Final step: Add shortcut to main path.
#         
#         x += x_shortcut
#         #x = layers.Add()([x, x_shortcut])
# =============================================================================
        return tf.nn.relu(x)
        
        
class ResNet1D(EmptyModel):
    """
    Instantiates the ResNet50 architecture in 1D similar to the original 
    ResNet paper.
    
    Implementation of the popular ResNet50 the following architecture:
    CONV1D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 ->
    CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> 
    IDBLOCK*2 -> AVGPOOL -> OUTPUTLAYER
    """
    def __init__(self, inputshape,
                 num_classes,
                 no_of_inputs=1):
        
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = no_of_inputs
        
        self.input_1 = layers.Input(shape=inputshape)
    
        # Zero-Padding
        self.zero_pad_1 = layers.ZeroPadding1D(padding=3)(self.input_1)
        
        # Stage 1
        self.conv_1 = layers.Conv1D(filters=12,
                                    kernel_size=1,
                                    strides=2,
                                    padding='valid',
                                    kernel_initializer=glorot_uniform(seed=0),
                                    name='stage_1_conv')(self.zero_pad_1)
        self.batch_1 = layers.BatchNormalization(
            axis=1,
            name='stage1_bn')(self.conv_1)
        self.act_1 = layers.Activation(activation='relu',
                                       name='stage_1_act')(self.batch_1)
        self.max_pool_1 = layers.MaxPooling1D(
            pool_size=3,
            strides=2,
            name='stage_1_max_pool')(self.act_1)
        
        # Stage 2
        self.conv_block_2a = ConvBlock(filters=[6,6,12],
                                       kernel_size_2=3,
                                       strides=2,
                                       stage=2,
                                       block='a')(self.max_pool_1)
        self.id_block_2b = IdentityBlock(filters=[6,6,12],
                                         kernel_size_2=3,
                                         stage=2,
                                         block='b')(self.conv_block_2a)
        self.id_block_2c = IdentityBlock(filters=[6,6,12],
                                         kernel_size_2=3,
                                         stage=2,
                                         block='c')(self.id_block_2b)

        # Stage 3
        self.conv_block_3a = ConvBlock(filters=[20,20,20],
                                       kernel_size_2=3,
                                       strides=2,
                                       stage=3,
                                       block='a')(self.id_block_2c)
        self.id_block_3b = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=3,
                                         block='b')(self.conv_block_3a)
        self.id_block_3c = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=3,
                                         block='c')(self.id_block_3b)  
        self.id_block_3d = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=3,
                                         block='d')(self.id_block_3c) 
        
        # Stage 4
        self.conv_block_4a = ConvBlock(filters=[20,20,20],
                                       kernel_size_2=3,
                                       strides=2,
                                       stage=4,
                                       block='a')(self.id_block_3d)
        self.id_block_4b = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='b')(self.conv_block_4a)
        self.id_block_4c = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='c')(self.id_block_4b)  
        self.id_block_4d = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='d')(self.id_block_4c)  
        self.id_block_4e = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='e')(self.id_block_4d)  
        self.id_block_4f = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='f')(self.id_block_4e)  

        # Stage 5
        self.conv_block_5a = ConvBlock(filters=[20,20,20],
                                       kernel_size_2=3,
                                       strides=2,
                                       stage=5,
                                       block='a')(self.id_block_4f)
        self.id_block_5b = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=5,
                                         stage=5,
                                         block='b')(self.conv_block_5a)
        self.id_block_5c = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=5,
                                         stage=5,
                                         block='c')(self.id_block_5b) 

        # Average pooling
        self.avg_pool = layers.AveragePooling1D(
            pool_size=2,
            name='avg_pool')(self.id_block_5c)
        
        # output layer
        self.flatten = layers.Flatten(name='flatten')(self.avg_pool)
        self.dense = layers.Dense(units=num_classes,
                                  activation='sigmoid',
                                  kernel_initializer = glorot_uniform(seed=0),
                                  name='dense')(self.flatten)
        
        self.output_norm = layers.Lambda(
            lambda x: x/tf.reshape(K.sum(x, axis=-1),(-1,1)),
            name = 'output_norm')(self.dense)

        super(ResNet1D, self).__init__(inputs=self.input_1,
                                       outputs=self.output_norm,
                                       inputshape=inputshape,
                                       num_classes=num_classes,
                                       name='ResNet1D')

class ResNet1DNew(models.Model):
    """
    Instantiates the ResNet50 architecture in 1D similar to the original 
    ResNet paper.
    
    Implementation of the popular ResNet50 the following architecture:
    CONV1D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 ->
    CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> 
    IDBLOCK*2 -> AVGPOOL -> OUTPUTLAYER
    """
    def __init__(self, inputshape,
                 num_classes,
                 no_of_inputs=1):
        
        super(ResNet1DNew, self).__init__(name='ResNet50')
        
        #self.input_1 = layers.Input(shape=inputshape)
    
        # Zero-Padding
        self.zero_pad_1 = layers.ZeroPadding1D(padding=3)
        
        # Stage 1
        self.conv_1 = layers.Conv1D(filters=12,
                                    kernel_size=1,
                                    strides=2,
                                    padding='valid',
                                    kernel_initializer=glorot_uniform(seed=0),
                                    name='stage1_conv')
        self.batch_1 = layers.BatchNormalization(
            axis=1,
            name='stage1_bn')
        self.act_1 = layers.Activation(activation='relu',
                                       name='stage1_act')
        self.max_pool_1 = layers.MaxPooling1D(pool_size=3,
                                              strides=2,
                                              name='stage1_max_pool')
        
        # Stage 2
        self.conv_block_2a = ConvBlock(filters=[6,6,12],
                                       kernel_size_2=3,
                                       stage=2,
                                       block='a',
                                       strides=2)
        self.id_block_2b = IdentityBlock(filters=[6,6,12],
                                         kernel_size_2=3,
                                         stage=2,
                                         block='b')
        self.id_block_2c = IdentityBlock(filters=[6,6,12],
                                         kernel_size_2=3,
                                         stage=2,
                                         block='c')

        # Stage 3
        self.conv_block_3a = ConvBlock(filters=[20,20,20],
                                       kernel_size_2=3,
                                       stage=3,
                                       block='a',
                                       strides=2)
        self.id_block_3b = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=3,
                                         block='b')
        self.id_block_3c = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=3,
                                         block='c') 
        self.id_block_3d = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=3,
                                         block='d')
        
        # Stage 4
        self.conv_block_4a = ConvBlock(filters=[20,20,20],
                                       kernel_size_2=3,
                                       stage=4,
                                       block='a',
                                       strides=2)
        self.id_block_4b = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='b')
        self.id_block_4c = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                          block='c') 
        self.id_block_4d = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='d') 
        self.id_block_4e = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='e') 
        self.id_block_4f = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=3,
                                         stage=4,
                                         block='f')  

        # Stage 5
        self.conv_block_5a = ConvBlock(filters=[20,20,20],
                                       kernel_size_2=3,
                                       stage=5,
                                       block='a',
                                       strides=2)
        self.id_block_5b = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=5,
                                         stage=3,
                                         block='b')
        self.id_block_5c = IdentityBlock(filters=[20,20,20],
                                         kernel_size_2=5,
                                         stage=3,
                                         block='c')

        # Average pooling
        self.avg_pool = layers.AveragePooling1D(
            pool_size=2,
            name='avg_pool')
        
        # output layer
        self.flatten = layers.Flatten(name='flatten')
        self.dense = layers.Dense(units=num_classes,
                                  activation='sigmoid',
                                  kernel_initializer = glorot_uniform(seed=0),
                                  name='dense_')
        
        # output norm
        self.output_norm = layers.Lambda(
            lambda x: x/tf.reshape(K.sum(x, axis=-1),(-1,1)),
            name = 'output_norm')

    def call(self, inputs, training=True):
        x = self.zero_pad_1(inputs)
        x = self.conv_1(x)
        x = self.batch_1(x, training=training)        
        x = self.act_1(x)
        
        x = self.max_pool_1(x)
        
        ### ResNet blocks ###
        resnet_blocks = [self.conv_block_2a,
                         self.id_block_2b,
                         self.id_block_2c,
                         self.conv_block_3a,
                         self.id_block_3b,
                         self.id_block_3c,
                         self.id_block_3d,
                         self.conv_block_4a,
                         self.id_block_4b,
                         self.id_block_4c,
                         self.id_block_4d,
                         self.id_block_4e,
                         self.id_block_4f,
                         self.conv_block_5a,
                         self.id_block_5b,
                         self.id_block_5c]                         
        
        for block in resnet_blocks:
            for layer in block:
                x = layer(x, training=training)  
# =============================================================================
#         # Stage 2
#         x = self.conv_block_2a(x, training=training)
#         x = self.id_block_2b(x, training=training)
#         x = self.id_block_2c(x, training=training)
#         
#         # Stage 3
#         x = self.conv_block_3a(x, training=training)
#         x = self.id_block_3b(x, training=training)
#         x = self.id_block_3c(x, training=training)
#         x = self.id_block_3d(x, training=training)
#         
#         # Stage 4
#         x = self.conv_block_4a(x, training=training)
#         x = self.id_block_4b(x, training=training)
#         x = self.id_block_4c(x, training=training)
#         x = self.id_block_4d(x, training=training)
#         x = self.id_block_4e(x, training=training)
#         x = self.id_block_4f(x, training=training)
#         
#         # Stage 5
#         x = self.conv_block_5a(x, training=training)
#         x = self.id_block_5b(x, training=training)
#         x = self.id_block_5c(x, training=training)
# =============================================================================
        
        # Average pooling
        x = self.avg_pool(x)
        
        # output layer
        x = self.flatten(x)
        x = self.dense(x)
        
        # output norm
        x = self.output_norm(x)

#%% 
if __name__ == "__main__":
    input_shape = (1121,1)
    num_classes = 4
    model = ResNet1D(input_shape,num_classes)
    #model = CustomCNNMultiple(input_shape,num_classes)
    model.build(input_shape)
    model.compile()
# =============================================================================
#     conv = ConvBlock(filters=[20,20,20],
#                               kernel_size_2=3,
#                               stage=5,
#                               block='a',
#                               strides=2)
#     conv.build(input_shape)
# =============================================================================
    model.summary()
        
# =============================================================================
#     import matplotlib.pyplot as plt
#     from tensorflow.keras.utils import plot_model
#     fig_file_name = r'C:\Users\pielsticker\Downloads\test_model.png'
#     plot_model(model, to_file = fig_file_name,
#                rankdir = "LR", show_shapes = True,
#                show_layer_names = True)
#     model_plot = plt.imread(fig_file_name)
#     fig, ax = plt.subplots(figsize=(18, 2))
#     ax.imshow(model_plot, interpolation='nearest')
#     plt.tight_layout()
#     plt.show()
# =============================================================================
