#!/usr/bin/env python3
"""
module for task 1
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds inception network based on paper
    """
    indata = K.Input(shape=(224, 224, 3))
    L1 = K.layers.Conv2D(64, 7, 2, activation='relu', padding='same')(indata)
    L2 = K.layers.MaxPool2D(3, 2, padding='same')(L1)
    # L3 = K.layers.Conv2D(64, 1, 1, activation='relu', padding='same')(L2)
    L4 = K.layers.Conv2D(192, 3, 1, activation='relu', padding='same')(L2)
    L5 = K.layers.MaxPool2D(3, 2, padding='same')(L4)
    L6 = inception_block(L5, [64, 96, 128, 16, 32, 32])
    L7 = inception_block(L6, [128, 128, 192, 32, 96, 64])
    L8 = K.layers.MaxPool2D(3, 2, padding='same')(L7)
    L9 = inception_block(L8, [192, 96, 208, 16, 48, 64])
    L10 = inception_block(L9, [160, 112, 224, 24, 64, 64])
    L11 = inception_block(L10, [128, 128, 256, 24, 64, 64])
    L12 = inception_block(L11, [112, 144, 288, 32, 64, 64])
    L13 = inception_block(L12, [256, 160, 320, 32, 128, 128])
    L14 = K.layers.MaxPool2D(3, 2, padding='same')(L13)
    L15 = inception_block(L14, [256, 160, 320, 32, 128, 128])
    L16 = inception_block(L15, [384, 192, 384, 48, 128, 128])
    L17 = K.layers.AvgPool2D(7, 1)(L16)
    L18 = K.layers.Dropout(.4)(L17)
    L19 = K.layers.Dense(1000)(L18)
    return K.Model(indata, L19)
