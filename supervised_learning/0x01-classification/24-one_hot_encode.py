#!/usr/bin/env python3
"""
module for one hot function
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    encodes labels vector into one-hot matrix
    """
    return np.eye(classes)[Y].T
