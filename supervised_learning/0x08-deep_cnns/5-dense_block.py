#!/usr/bin/env python3
"""
module for task 5
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds dense block based on paper
    """
    prev = X
    for i in range(layers):
        L1 = K.layers.BatchNormalization()(prev)
        L2 = K.layers.Activation('relu')(L1)
        L3 = K.layers.Conv2D(
            growth_rate * 4, 1, kernel_initializer='he_normal')(L2)
        L4 = K.layers.BatchNormalization()(L3)
        L5 = K.layers.Activation('relu')(L4)
        L6 = K.layers.Conv2D(
            growth_rate, 3, padding='same', kernel_initializer='he_normal')(L5)
        prev = K.layers.concatenate([prev, L6])
    return prev, nb_filters + (growth_rate * layers)
