#!/usr/bin/env python3
"""
module for task 5
"""

import tensorflow.keras as K


def lenet5(X):
    """
    builds a version of LeNet-5 using keras
    """
    ki = K.initializers.he_normal()
    L1 = K.layers.Conv2D(filters=6, kernel_size=(
        5, 5), padding='same', activation=K.activations.relu,
                         kernel_initializer=ki)(X)
    L2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(L1)
    L3 = K.layers.Conv2D(filters=16, kernel_size=(
        5, 5), padding='valid', activation=K.activations.relu,
                         kernel_initializer=ki)(L2)
    L4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(L3)
    out = K.layers.Flatten()(L4)
    L5 = K.layers.Dense(
        120, activation=K.activations.relu, kernel_initializer=ki)(out)
    L6 = K.layers.Dense(
        84, activation=K.activations.relu, kernel_initializer=ki)(L5)
    L7 = K.layers.Dense(10, kernel_initializer=ki)(L6)
    out = K.layers.Softmax()(L7)
    model = K.Model(inputs=X, outputs=out)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
