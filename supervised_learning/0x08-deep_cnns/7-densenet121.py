#!/usr/bin/env python3
"""
module for task 7
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds DenseNet-121 architecture
    """
    indata = K.Input((224, 224, 3))
    L1 = K.layers.BatchNormalization()(indata)
    L2 = K.layers.Activation('relu')(L1)
    L3 = K.layers.Conv2D(growth_rate * 2, 7, 2,
                         padding='same', kernel_initializer='he_normal')(L2)
    L4 = K.layers.MaxPool2D(2)(L3)
    L5, nb_filters = dense_block(L4, growth_rate * 2, growth_rate, 6)
    L6, nb_filters = transition_layer(L5, nb_filters, compression)
    L7, nb_filters = dense_block(L6, nb_filters, growth_rate, 12)
    L8, nb_filters = transition_layer(L7, nb_filters, compression)
    L9, nb_filters = dense_block(L8, nb_filters, growth_rate, 24)
    L10, nb_filters = transition_layer(L9, nb_filters, compression)
    L11, nb_filters = dense_block(L10, nb_filters, growth_rate, 16)
    L12 = K.layers.AvgPool2D(7)(L11)
    L13 = K.layers.Dense(
        1000, kernel_initializer='he_normal', activation='softmax')(L12)
    return K.Model(indata, L13)
