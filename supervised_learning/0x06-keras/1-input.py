#!/usr/bin/env python3
"""
module for task 1
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network
    """
    inputs = K.Input(shape=(nx,))
    l2 = K.regularizers.l2(lambtha)
    x = K.layers.Dense(
        layers[0], activation=activations[0], kernel_regularizer=l2)(inputs)
    for layer in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[layer], activation=activations[layer],
            kernel_regularizer=l2)(x)
    return K.models.Model(inputs=inputs, outputs=x)
