#!/usr/bin/env python3
"""
module for task 3
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a lbal into a one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
