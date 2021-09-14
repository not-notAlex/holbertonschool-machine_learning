#!/usr/bin/env python3
"""
module for task 4
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds ResNet-50 architecture
    """
    indata = K.layers.Input((224, 224, 3))
    L1 = K.layers.Conv2D(
        64, 7, 2, padding='same', kernel_initializer='he_normal')(indata)
    L2 = K.layers.BatchNormalization()(L1)
    L3 = K.layers.Activation('relu')(L2)
    L4 = K.layers.MaxPool2D(3, 2, padding='same')(L3)
    L5 = projection_block(L4, [64, 64, 256], 1)
    L6 = identity_block(L5, [64, 64, 256])
    L7 = identity_block(L6, [64, 64, 256])
    L8 = projection_block(L7, [128, 128, 512], 2)
    L9 = identity_block(L8, [128, 128, 512])
    L10 = identity_block(L9, [128, 128, 512])
    L11 = identity_block(L10, [128, 128, 512])
    L12 = projection_block(L11, [256, 256, 1024], 2)
    L13 = identity_block(L12, [256, 256, 1024])
    L14 = identity_block(L13, [256, 256, 1024])
    L15 = identity_block(L14, [256, 256, 1024])
    L16 = identity_block(L15, [256, 256, 1024])
    L17 = identity_block(L16, [256, 256, 1024])
    L18 = projection_block(L17, [512, 512, 2048], 2)
    L19 = identity_block(L18, [512, 512, 2048])
    L20 = identity_block(L19, [512, 512, 2048])
    L21 = K.layers.AvgPool2D(7)(L20)
    L22 = K.layers.Dense(
        1000, kernel_initializer='he_normal', activation='softmax')(L21)
    return K.Model(indata, L22)
