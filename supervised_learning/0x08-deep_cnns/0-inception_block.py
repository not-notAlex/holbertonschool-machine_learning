#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds inception block based on paper
    """
    LF1, LF2, LF3, LF4, LF5, LF7 = filters
    L1 = K.layers.Conv2D(LF1, 1, activation='relu')(A_prev)
    L2 = K.layers.Conv2D(LF2, 1, activation='relu')(A_prev)
    L3 = K.layers.Conv2D(LF3, 3, padding='same', activation='relu')(L2)
    L4 = K.layers.Conv2D(LF4, 1, activation='relu')(A_prev)
    L5 = K.layers.Conv2D(LF5, 5, padding='same', activation='relu')(L4)
    L6 = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    L7 = K.layers.Conv2D(LF7, 1, activation='relu')(L6)
    return K.layers.concatenate([L1, L3, L5, L7])
