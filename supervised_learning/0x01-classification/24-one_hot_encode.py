#!/usr/bin/env python3
"""
module for one hot function
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    encodes labels vector into one-hot matrix
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int or classes < 2:
        return None
    return np.eye(classes)[Y].T
