# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:10:25 2020.

@author: pielsticker
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend as K

#%%
class EmptyModel(models.Model):
    """Base Model class."""

    def __init__(
        self,
        inputs,
        outputs,
        inputshape,
        num_classes,
        no_of_inputs=1,
        name="New_Model",
    ):
        """
        Intialize emppy keras model.

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

        super(EmptyModel, self).__init__(
            inputs=inputs, outputs=outputs, name=name
        )

    def get_config(self):
        """
        Overwrite get_config method.

        For serialization, all input paramters of the model are added to
        the get_config method from the keras.Model class.

        Returns
        -------
        config : dict
            Configuration of the model.

        """
        # For serialization with 'custom_objects'
        config = super(EmptyModel, self).get_config()
        config["inputshape"] = self.inputshape
        config["num_classes"] = self.num_classes
        config["no_of_inputs"] = self.no_of_inputs

        return config


class CustomMLP(EmptyModel):
    """A neural net with some hidden layers, but no convolutions."""

    def __init__(self, inputshape, num_classes):
        """
        Initialize model with dense and batch norm layers.

        Parameters
        ----------
        inputshape : ndarray
            Shape of the features of the training data set.
        num_classes : ndarray
            Shape of the labels of the training data set.

        Returns
        -------
        None.

        """
        self.input_1 = layers.Input(shape=inputshape, name="input_1")

        self.flatten_1 = layers.Flatten(name="flatten1")(self.input_1)
        self.drop_1 = layers.Dropout(rate=0.5, name="drop_1")(self.flatten_1)
        self.dense_1 = layers.Dense(
            units=64, activation="relu", name="dense1"
        )(self.drop_1)
        self.batch_norm_1 = layers.BatchNormalization(name="batch_norm_1")(
            self.dense_1
        )
        self.dense_2 = layers.Dense(
            units=num_classes, activation="softmax", name="dense2"
        )(self.batch_norm_1)

        super(CustomMLP, self).__init__(
            inputs=self.input_1,
            outputs=self.dense_2,
            inputshape=inputshape,
            num_classes=num_classes,
            name="Custom_MLP",
        )


class ClassificationCNN(EmptyModel):
    """A CNN for XPS data with only one phase."""

    def __init__(self, inputshape, num_classes):
        """
        Initialize model with convolutional layers.

        A CNN with three convolutional layers of different kernel size
        at the beginning. Works well for learning across scales.
        This is to be used for classification -> softmax activation in
        the last layer.

        Parameters
        ----------
        inputshape : ndarray
            Shape of the features of the training data set.
        num_classes : ndarray
            Shape of the labels of the training data set.

        Returns
        -------
        None.

        """
        self.input_1 = layers.Input(shape=inputshape)

        self.conv_1_short = layers.Conv1D(
            filters=4,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_short",
        )(self.input_1)
        self.conv_1_medium = layers.Conv1D(
            filters=4,
            kernel_size=10,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_medium",
        )(self.input_1)
        self.conv_1_long = layers.Conv1D(
            filters=4,
            kernel_size=15,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_long",
        )(self.input_1)

        sublayers = [
            self.conv_1_short,
            self.conv_1_medium,
            self.conv_1_long,
        ]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = layers.Conv1D(
            filters=4,
            kernel_size=10,
            strides=1,
            padding="valid",
            activation="relu",
            name="conv_2",
        )(merged_sublayers)
        self.conv_3 = layers.Conv1D(
            filters=10,
            kernel_size=10,
            strides=1,
            padding="valid",
            activation="relu",
            name="conv_3",
        )(self.conv_2)
        self.average_pool_1 = layers.AveragePooling1D(name="average_pool_1")(
            self.conv_3
        )

        self.flatten_1 = layers.Flatten(name="flatten1")(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2, name="drop_1")(self.flatten_1)
        self.dense_1 = layers.Dense(
            units=1000, activation="relu", name="dense_1"
        )(self.drop_1)
        self.dense_2 = layers.Dense(
            units=num_classes, activation="softmax", name="dense_2"
        )(self.dense_1)

        no_of_inputs = len(sublayers)

        super(ClassificationCNN, self).__init__(
            inputs=self.input_1,
            outputs=self.dense_2,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="ClassificationCNN",
        )


class RegressionCNN(EmptyModel):
    """A CNN for XPS data with mixed phases."""

    def __init__(self, inputshape, num_classes):
        """
        Initialize model with convolutional layers.

        A CNN with three convolutional layers of different kernel size
        at the beginning. Works well for learning across scales.

        This is to be used for regression on all labels. -> sigmoid
        activation in the last layer.
        """
        self.input_1 = layers.Input(shape=inputshape)

        self.conv_1_short = layers.Conv1D(
            filters=12,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_short",
        )(self.input_1)
        self.conv_1_medium = layers.Conv1D(
            filters=12,
            kernel_size=10,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_medium",
        )(self.input_1)
        self.conv_1_long = layers.Conv1D(
            filters=12,
            kernel_size=15,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_long",
        )(self.input_1)

        sublayers = [
            self.conv_1_short,
            self.conv_1_medium,
            self.conv_1_long,
        ]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = layers.Conv1D(
            filters=10,
            kernel_size=5,
            strides=1,
            padding="valid",
            activation="relu",
            name="conv_2",
        )(merged_sublayers)
        self.conv_3 = layers.Conv1D(
            filters=10,
            kernel_size=5,
            strides=1,
            padding="valid",
            activation="relu",
            name="conv_3",
        )(self.conv_2)
        self.average_pool_1 = layers.AveragePooling1D(name="average_pool_1")(
            self.conv_3
        )

        self.flatten_1 = layers.Flatten(name="flatten1")(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2, name="drop_1")(self.flatten_1)
        self.dense_1 = layers.Dense(
            units=4000, activation="relu", name="dense_1"
        )(self.drop_1)
        self.dense_2 = layers.Dense(
            units=num_classes, activation="sigmoid", name="dense_2"
        )(self.dense_1)

        self.output_norm = layers.Lambda(
            lambda x: x / tf.reshape(K.sum(x, axis=-1), (-1, 1)),
            name="output_normalization",
        )(self.dense_2)

        no_of_inputs = len(sublayers)

        super(RegressionCNN, self).__init__(
            inputs=self.input_1,
            outputs=self.output_norm,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="RegressionCNN",
        )


class ClassificationCNN2D(EmptyModel):
    """
    A CNN with three convolutional layers of different kernel size at
    the beginning. Works well for learning across scales.

    This is to be used for classification -> softmax activation in the
    last layer.

    2D model for e.g. MNIST.
    """

    def __init__(self, inputshape, num_classes):

        self.input_1 = layers.Input(shape=inputshape)

        self.conv_1_short = layers.Conv2D(
            filters=4,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            name="conv_1_short",
        )(self.input_1)
        self.conv_1_medium = layers.Conv2D(
            filters=4,
            kernel_size=10,
            padding="same",
            activation="relu",
            name="conv_1_medium",
        )(self.input_1)
        self.conv_1_long = layers.Conv2D(
            filters=4,
            kernel_size=15,
            padding="same",
            activation="relu",
            name="conv_1_long",
        )(self.input_1)

        sublayers = [
            self.conv_1_short,
            self.conv_1_medium,
            self.conv_1_long,
        ]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = layers.Conv2D(
            filters=4,
            kernel_size=5,
            padding="valid",
            activation="relu",
            name="conv_2",
        )(merged_sublayers)
        self.average_pool_1 = layers.AveragePooling2D(name="average_pool_1")(
            self.conv_2
        )

        self.flatten_1 = layers.Flatten(name="flatten1")(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2, name="drop_1")(self.flatten_1)
        self.dense_1 = layers.Dense(
            units=1000, activation="relu", name="dense_1"
        )(self.drop_1)
        self.dense_2 = layers.Dense(
            units=num_classes, activation="softmax", name="dense_2"
        )(self.dense_1)

        no_of_inputs = len(sublayers)

        super(ClassificationCNN2D, self).__init__(
            inputs=self.input_1,
            outputs=self.dense_2,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="ClassificationCNN2D",
        )


### RESNET50 implementation ###
class IdentityBlock(models.Model):
    """Model implementing the IdentityBlock in ResNets."""

    def __init__(self, filters, kernel_size_2, stage, block, strides=1):
        """
        Initialize the layers.

        IdentityBlock contain a main path (3 convolutional layers with
        subsequent batch norm layers) and a shortcut path
        without convolutional layers.

        Parameters
        ----------
        filters : tuple
            Filter sizes for the 3 convolutional layer at main path.
            The third filter is used in the shortcut path, too.
        kernel_size_2 : int
            The kernel size of the middle convolutional layer at
            main path.
        stage : int
            Current stage label, used for generating layer names.
            1, 2, ...
        block : str
            Current block label, used for generating layer names.
            "a", "b", ...
        strides : int , optional
            Strides for the first convolutional layer in the block.
            The default is 1.

        Returns
        -------
        None.

        """
        name = str(stage) + str(block) + "_ID"
        super(IdentityBlock, self).__init__(name=name)

        # Store filters
        filter1, filter2, filter3 = filters

        ### Main Path ###
        # Component 1
        self.conv_1 = layers.Conv1D(
            filters=filter1,
            kernel_size=1,
            strides=strides,
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv1",
        )
        self.batch_1 = layers.BatchNormalization(axis=1, name=name + "_bn1")
        # Component 2
        self.conv_2 = layers.Conv1D(
            filters=filter2,
            kernel_size=kernel_size_2,
            strides=strides,
            padding="same",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv2",
        )

        self.batch_2 = layers.BatchNormalization(axis=1, name=name + "_bn2")

        # Component 3
        self.conv_3 = layers.Conv1D(
            filters=filter3,
            kernel_size=1,
            strides=strides,
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv3",
        )
        self.batch_3 = layers.BatchNormalization(axis=1, name=name + "_bn3")

    def call(self, x, training=False):
        """
        Call the model on new inputs.

        There is no convolution in the shortcut path.

        Parameters
        ----------
        x : A tensor or list of tensors.
            Inputs for the forward pass.
        training : bool, optional
            Boolean or boolean scalar tensor, indicating whether to run
            the model in training mode or inference mode.

        Returns
        -------
        tensor or list of tensors
            Model output.

        """
        # Intermediate save of the input.
        x_shortcut = x

        ### Main Path ###
        x = self.conv_1(x)
        x = self.batch_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.batch_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_3(x)
        x = self.batch_3(x, training=training)

        # Final step: Add shortcut to main path.
        x += x_shortcut
        return tf.nn.relu(x)


class ConvBlock(models.Model):
    """Model implementing the ConvBlock in ResNets."""

    def __init__(self, filters, kernel_size_2, stage, block, strides=2):
        """
        Initialize the layers.

        ConvBlocks contain a main path (3 convolutional layers with
        subsequent batch norm layers) and a shortcut path
        (3 convolutional layers with subsequent batch norm layers).

        Parameters
        ----------
        filters : tuple
            Filter sizes for the 3 convolutional layer at main path.
            The third filter is used in the shortcut path, too.
        kernel_size_2 : int
            The kernel size of the middle convolutional layer at
            main path.
        stage : int
            Current stage label, used for generating layer names.
            1, 2, ...
        block : str
            Current block label, used for generating layer names.
            "a", "b", ...
        strides : int , optional
            Strides for the first convolutional layer in the block.
            The default is 2.

        Returns
        -------
        None.

        """
        name = str(stage) + str(block) + "_CONV"
        super(ConvBlock, self).__init__(name=name)

        # Store filters
        filter1, filter2, filter3 = filters

        ### Main Path ###
        # Component 1
        self.conv_1 = layers.Conv1D(
            filters=filter1,
            kernel_size=1,
            strides=strides,
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv1",
        )
        self.batch_1 = layers.BatchNormalization(axis=1, name=name + "_bn1")
        # Component 2
        self.conv_2 = layers.Conv1D(
            filters=filter2,
            kernel_size=kernel_size_2,
            strides=1,
            padding="same",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv2",
        )

        self.batch_2 = layers.BatchNormalization(axis=1, name=name + "_bn2")

        # Component 3
        self.conv_3 = layers.Conv1D(
            filters=filter3,
            kernel_size=2,
            strides=1,
            padding="same",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv3",
        )
        self.batch_3 = layers.BatchNormalization(axis=1, name=name + "_bn3")

        ### Shortcut Path ###
        self.conv_short = layers.Conv1D(
            filters=filter3,
            kernel_size=2,
            strides=strides,
            padding="same",
            kernel_initializer=glorot_uniform(seed=0),
            name=name + "_conv_short",
        )
        self.batch_short = layers.BatchNormalization(
            axis=1, name=name + "_bn_short"
        )

    def call(self, inputs, training=False):
        """
        Call the model on new inputs.

        At shortcut, there is a convolutional layer.

        Parameters
        ----------
        x : A tensor or list of tensors.
            Inputs for the forward pass.
        training : bool, optional
            Boolean or boolean scalar tensor, indicating whether to run
            the model in training mode or inference mode.

        Returns
        -------
        tensor or list of tensors
            Model output.

        """
        # Intermediate save of the input.
        x = inputs
        x_shortcut = inputs

        ### Main Path ###
        x = self.conv_1(x)
        x = self.batch_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.batch_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv_3(x)
        x = self.batch_3(x, training=training)

        ### Shortcut Path ###
        x_shortcut = self.conv_short(x_shortcut)
        x_shortcut = self.batch_short(x_shortcut)

        ### Final step: Add shortcut to main path ###
        x += x_shortcut

        return tf.nn.relu(x)


class ResNet1D(EmptyModel):
    """Class instantiatingthe ResNet50 architecture in 1D."""

    def __init__(self, inputshape, num_classes, ap=False, no_of_inputs=1):
        """
        Instantiate layers.

        This implementation uses the functional API in Keras.
        Architecture:
            CONV1D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK
            -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK
            -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2
            -> AVGPOOL (optional) -> OUTPUTLAYER
            -> NORMALIZED OUTPUTLAYER

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        ap : bool, optional
            If ap, then an AveragePooling1D layer is added after the
            residual blocks. The default is False.
        no_of_inputs : int, optional
            Number of input layers. The default is 1.
            (not working here)

        Returns
        -------
        None.

        """
        self.ap = ap

        self.input_1 = layers.Input(shape=inputshape, name="input_1")

        # Zero-Padding
        self.zero_pad_1 = layers.ZeroPadding1D(padding=3)(self.input_1)

        # Stage 1
        self.conv_1 = layers.Conv1D(
            filters=64,
            kernel_size=2,
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
            name="stage1_conv",
        )(self.zero_pad_1)
        self.batch_1 = layers.BatchNormalization(axis=1, name="stage1_bn")(
            self.conv_1
        )
        self.act_1 = layers.Activation(activation="relu", name="stage1_act")(
            self.batch_1
        )
        self.max_pool_1 = layers.MaxPooling1D(
            pool_size=1, strides=1, name="stage1_max_pool"
        )(self.act_1)

        # Stage 2
        self.conv_block_2a = ConvBlock(
            filters=[32, 32, 128], kernel_size_2=4, stage=2, block="a"
        )(self.max_pool_1)
        self.id_block_2b = IdentityBlock(
            filters=[32, 32, 128], kernel_size_2=1, stage=2, block="b"
        )(self.conv_block_2a)
        self.id_block_2c = IdentityBlock(
            filters=[32, 32, 128], kernel_size_2=1, stage=2, block="c"
        )(self.id_block_2b)

        # Stage 3
        self.conv_block_3a = ConvBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="a"
        )(self.id_block_2c)
        self.id_block_3b = IdentityBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="b"
        )(self.conv_block_3a)
        self.id_block_3c = IdentityBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="c"
        )(self.id_block_3b)
        self.id_block_3d = IdentityBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="d"
        )(self.id_block_3c)

        # Stage 4
        self.conv_block_4a = ConvBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="a",
        )(self.id_block_3d)
        self.id_block_4b = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="b",
        )(self.conv_block_4a)
        self.id_block_4c = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="c",
        )(self.id_block_4b)
        self.id_block_4d = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="d",
        )(self.id_block_4c)
        self.id_block_4e = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="e",
        )(self.id_block_4d)
        self.id_block_4f = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="f",
        )(self.id_block_4e)

        # Stage 5
        self.conv_block_5a = ConvBlock(
            filters=[256, 56, 1024], kernel_size_2=1, stage=5, block="a"
        )(self.id_block_4f)
        self.id_block_5b = IdentityBlock(
            filters=[256, 56, 1024], kernel_size_2=1, stage=5, block="b"
        )(self.conv_block_5a)
        self.id_block_5c = IdentityBlock(
            filters=[256, 56, 1024], kernel_size_2=1, stage=5, block="c"
        )(self.id_block_5b)

        # Average pooling
        if self.ap:
            self.avg_pool = layers.AveragePooling1D(
                pool_size=3, name="avg_pool"
            )(self.id_block_5c)
            self.flatten = layers.Flatten(name="flatten")(self.avg_pool)

        else:
            self.flatten = layers.Flatten(name="flatten")(self.id_block_5c)

        # output layer
        self.dense = layers.Dense(
            units=num_classes,
            activation="sigmoid",
            kernel_initializer=glorot_uniform(seed=0),
            name="dense",
        )(self.flatten)

        # output norm
        self.output_norm = layers.Lambda(
            lambda x: x / tf.reshape(K.sum(x, axis=-1), (-1, 1)),
            name="output_norm",
        )(self.dense)

        super(ResNet1D, self).__init__(
            inputs=self.input_1,
            outputs=self.output_norm,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="ResNet1D",
        )


class ResNet1DSubclassed(models.Model):
    """Class instantiatingthe ResNet50 architecture in 1D."""

    def __init__(self, num_classes, ap=False, no_of_inputs=1):
        """
        Instantiate layers.

        This implementation uses subclassing of models.
        Architecture:
            CONV1D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK
            -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK
            -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2
            -> AVGPOOL (optional) -> OUTPUTLAYER
            -> NORMALIZED OUTPUTLAYER
        Parameters
        ----------
        num_classes : int
            Number of output classes.
        ap : bool, optional
            If ap, then an AveragePooling1D layer is added after the
            residual blocks. The default is False.
        no_of_inputs : int, optional
            Number of input layers. The default is 1.
            (not working here)

        Returns
        -------
        None.

        """
        self.ap = ap

        super(ResNet1DSubclassed, self).__init__(name="ResNet1D")

        # Zero-Padding
        self.zero_pad_1 = layers.ZeroPadding1D(padding=3)

        # Stage 1
        self.conv_1 = layers.Conv1D(
            filters=64,
            kernel_size=2,
            padding="valid",
            kernel_initializer=glorot_uniform(seed=0),
            name="stage1_conv",
        )
        self.batch_1 = layers.BatchNormalization(axis=1, name="stage1_bn")
        self.act_1 = layers.Activation(activation="relu", name="stage1_act")
        self.max_pool_1 = layers.MaxPooling1D(
            pool_size=1, strides=1, name="stage1_max_pool"
        )

        # Stage 2
        self.conv_block_2a = ConvBlock(
            filters=[32, 32, 128], kernel_size_2=4, stage=2, block="a"
        )
        self.id_block_2b = IdentityBlock(
            filters=[32, 32, 128], kernel_size_2=1, stage=2, block="b"
        )
        self.id_block_2c = IdentityBlock(
            filters=[32, 32, 128], kernel_size_2=1, stage=2, block="c"
        )

        # Stage 3
        self.conv_block_3a = ConvBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="a"
        )
        self.id_block_3b = IdentityBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="b"
        )
        self.id_block_3c = IdentityBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="c"
        )
        self.id_block_3d = IdentityBlock(
            filters=[64, 64, 256], kernel_size_2=3, stage=3, block="d"
        )

        # Stage 4
        self.conv_block_4a = ConvBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="a",
        )
        self.id_block_4b = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="b",
        )
        self.id_block_4c = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="c",
        )
        self.id_block_4d = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="d",
        )
        self.id_block_4e = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="e",
        )
        self.id_block_4f = IdentityBlock(
            filters=[128, 128, 1024],
            kernel_size_2=3,
            stage=4,
            block="f",
        )

        # Stage 5
        self.conv_block_5a = ConvBlock(
            filters=[256, 56, 1024], kernel_size_2=1, stage=5, block="a"
        )
        self.id_block_5b = IdentityBlock(
            filters=[256, 56, 1024], kernel_size_2=1, stage=5, block="b"
        )
        self.id_block_5c = IdentityBlock(
            filters=[256, 56, 1024], kernel_size_2=1, stage=5, block="c"
        )

        # Average pooling
        if self.ap:
            self.avg_pool = layers.AveragePooling1D(
                pool_size=3, name="avg_pool"
            )

        # output layer
        self.flatten = layers.Flatten(name="flatten")
        self.dense = layers.Dense(
            units=num_classes,
            activation="sigmoid",
            kernel_initializer=glorot_uniform(seed=0),
            name="dense",
        )

        # output norm
        self.output_norm = layers.Lambda(
            lambda x: x / tf.reshape(K.sum(x, axis=-1), (-1, 1)),
            name="output_norm",
        )

    def call(self, x, training=True):
        """
        Call the model on new inputs.

        Parameters
        ----------
        x : A tensor or list of tensors.
            Inputs for the forward pass.
        training : bool, optional
            Boolean or boolean scalar tensor, indicating whether to run
            the model in training mode or inference mode.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        x = self.zero_pad_1(x)
        x = self.conv_1(x)
        x = self.batch_1(x, training=training)
        x = self.act_1(x)
        x = self.max_pool_1(x)

        resnet_blocks = [
            self.conv_block_2a,
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
            self.id_block_5c,
        ]

        for block in resnet_blocks:
            x = block(x)

        # Average pooling
        if self.ap:
            x = self.avg_pool(x)

        # output layer
        x = self.flatten(x)
        x = self.dense(x)

        # output norm
        return self.output_norm(x)


### Probabilistic implementations ###
class BayesianClassificationCNN2D(EmptyModel):
    """
    A CNN with three convolutional layers of different kernel size at
    the beginning. Works well for learning across scales.

    This is to be used for classification -> softmax activation in the
    last layer.

    2D model for e.g. MNIST.
    Implements Bayes by Backprop.
    """

    def __init__(
        self,
        inputshape,
        num_classes,
        kl_divergence_function,
        bias_divergence_fn,
    ):

        self.input_1 = layers.Input(shape=inputshape)

        self.conv_1_short = tfp.layers.Convolution2DFlipout(
            filters=4,
            kernel_size=5,
            strides=1,
            padding="same",
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
            name="conv_1_short",
        )(self.input_1)
        self.conv_1_medium = tfp.layers.Convolution2DFlipout(
            filters=4,
            kernel_size=10,
            padding="same",
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
            name="conv_1_medium",
        )(self.input_1)
        self.conv_1_long = tfp.layers.Convolution2DFlipout(
            filters=4,
            kernel_size=15,
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=bias_divergence_fn,
            padding="same",
            activation="relu",
            name="conv_1_long",
        )(self.input_1)

        sublayers = [
            self.conv_1_short,
            self.conv_1_medium,
            self.conv_1_long,
        ]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = tfp.layers.Convolution2DFlipout(
            filters=4,
            kernel_size=5,
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=bias_divergence_fn,
            padding="valid",
            activation="relu",
            name="conv_2",
        )(merged_sublayers)

        self.average_pool_1 = layers.AveragePooling2D(name="average_pool_1")(
            self.conv_2
        )

        self.flatten_1 = layers.Flatten(name="flatten1")(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2, name="drop_1")(self.flatten_1)
        self.dense_1 = tfp.layers.DenseFlipout(
            units=1000,
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=bias_divergence_fn,
            activation="relu",
            name="dense_1",
        )(self.drop_1)
        self.dense_2 = tfp.layers.DenseFlipout(
            units=num_classes,
            kernel_divergence_fn=kl_divergence_function,
            bias_divergence_fn=bias_divergence_fn,
            activation="softmax",
            name="dense_2",
        )(self.dense_1)

        no_of_inputs = len(sublayers)

        super(BayesianClassificationCNN2D, self).__init__(
            inputs=self.input_1,
            outputs=self.dense_2,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="BayesianClassificationCNN2D",
        )


class BayesianCNN(EmptyModel):
    """
    A CNN with three convolutional layers of different kernel size at
    the beginning. Works well for learning across scales.

    This is to be used for regression on all labels. -> sigmoid
    activation in the last layer.
    """

    def __init__(
        self,
        inputshape,
        num_classes,
        kl_divergence_fn,
        task,
    ):
        if len(inputshape) == 2:
            conv_layer = tfp.layers.Convolution1DFlipout
            strides = 1
            average_pool_layer = layers.AveragePooling1D
        elif len(inputshape) == 3:
            conv_layer = tfp.layers.Convolution2DFlipout
            strides = (1, 1)
            average_pool_layer = layers.AveragePooling2D

        if task == "regression":
            if num_classes == 1:
                output_act = None
            else:
                output_act = None  # "sigmoid"
        elif task == "classification":
            output_act = "softmax"

        prob_act = "softplus"

        self.input_1 = layers.Input(shape=inputshape, name="input_1")
        self.conv_1_short = conv_layer(
            filters=12,
            kernel_size=5,
            strides=strides,
            padding="same",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,
            name="conv_1_short",
        )(self.input_1)
        self.conv_1_medium = conv_layer(
            filters=12,
            kernel_size=10,
            strides=strides,
            padding="same",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,
            name="conv_1_medium",
        )(self.input_1)
        self.conv_1_long = conv_layer(
            filters=12,
            kernel_size=15,
            strides=strides,
            padding="same",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,
            name="conv_1_long",
        )(self.input_1)

        sublayers = [
            self.conv_1_short,
            self.conv_1_medium,
            self.conv_1_long,
        ]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = conv_layer(
            filters=10,
            kernel_size=5,
            strides=strides,
            padding="valid",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,
            name="conv_2",
        )(merged_sublayers)
        self.conv_3 = conv_layer(
            filters=10,
            kernel_size=5,
            strides=strides,
            padding="valid",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,
            name="conv_3",
        )(self.conv_2)
        self.average_pool_1 = average_pool_layer(name="average_pool_1")(
            self.conv_3
        )

        self.flatten_1 = layers.Flatten(name="flatten1")(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2, name="drop_1")(self.flatten_1)
        self.dense_1 = tfp.layers.DenseFlipout(
            units=4000,
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,
            name="dense_1",
        )(self.flatten_1)

        self.dense_2 = tfp.layers.DenseFlipout(
            units=num_classes,
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=output_act,
            name="dense_2",
        )(self.dense_1)

        no_of_inputs = len(sublayers)

        super(BayesianCNN, self).__init__(
            inputs=self.input_1,
            outputs=self.dense_2,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="BayesianCNN",
        )


#%%
if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_shape = (1121, 2)
    num_classes = 4
    # model = ResNet1D(input_shape, num_classes)
    model = RegressionCNN(input_shape, num_classes)
    # model = ResNet1DSubclassed(num_classes)
    # model.build((1,1121,1))
    model.compile()
    # model.summary()
