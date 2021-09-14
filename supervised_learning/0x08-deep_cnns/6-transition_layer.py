#!/usr/bin/env python3
"""
module for task 6
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds transition layer based on paper
    """
    L1 = K.layers.BatchNormalization()(X)
    L2 = K.layers.Activation('relu')(L1)
    L3 = K.layers.Conv2D(int(
        nb_filters * compression), 1, kernel_initializer='he_normal')(L2)
    return K.layers.AvgPool2D(2)(L3), int(nb_filters * compression)
