#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a network with keras
    """
    model = K.Sequential()
    l2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=l2, input_shape=(nx,)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=l2, input_shape=(nx,)))
    return model
