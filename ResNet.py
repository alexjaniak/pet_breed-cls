"""
-*- defines ResNet model architecture -*-

@author:    alexjaniak
@date:      6/3/20
@file:      ResNet.py  
"""

# TODO: only include necessary inputs i.e from numpy import array
import tensorflow as tf
from tensorflow import keras 

def resnet34(output_nodes, input_shape=[224,224,3]):
    """
    builds model/layers according to the ResNet-34 architecture

    :param output_nodes: # of outputs, usually # of categorical vars
    :param input_shape: the input shape
    :return: ResNet-34 model
    """

    # primary layers
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=input_shape, padding="same",use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")) # take max of 3x3 kernel, halfs the size

    # add residual layers
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2 # decrease size if filters increase
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    # output layers
    model.add(keras.layers.GlobalAvgPool2D()) # 1 val per filter
    model.add(keras.layers.Flatten()) # resize for final output layer
    model.add(keras.layers.Dense(output_nodes, activation="softmax")) # output

    return model

class ResidualUnit(keras.layers.Layer):
    """ 
    Residual Unit (block of layers that sums input & output)
    """

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        """
        creates a Residual Unit instance

        :param filters: # of filters for conv2d layers
        :param strides: stride length
        :param activation: activation function/layer
        :param **kwargs: args for keras.laters.Layer object
        :return: returns nothing
        """
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]

        # fit shape of input to output if different sizes (strides > 1)
        self.skip_layers = [] 
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]
        
    def call(self, inputs):
        """
        main executable function for layer object

        :param inputs: layer inputs 
        :return: return sum of outputs & resized inputs
        """
        # get outputs
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        # resize inputs
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z) # sums input & output
