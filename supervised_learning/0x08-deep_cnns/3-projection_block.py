#!/usr/bin/env python3
"""
module for task 3
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    builds projection block based on paper
    """
    LF1, LF4, LF7 = filters
    L1 = K.layers.Conv2D(LF1, 1, s, kernel_initializer='he_normal')(A_prev)
    L2 = K.layers.BatchNormalization()(L1)
    L3 = K.layers.Activation('relu')(L2)
    L4 = K.layers.Conv2D(
        LF4, 3, padding='same', kernel_initializer='he_normal')(L3)
    L5 = K.layers.BatchNormalization()(L4)
    L6 = K.layers.Activation('relu')(L5)
    L7 = K.layers.Conv2D(LF7, 1, kernel_initializer='he_normal')(L6)
    L8 = K.layers.BatchNormalization()(L7)
    L9 = K.layers.Conv2D(LF7, 1, s, kernel_initializer='he_normal')(A_prev)
    L10 = K.layers.BatchNormalization()(L9)
    L11 = K.layers.add([L8, L10])
    return K.layers.Activation('relu')(L11)
