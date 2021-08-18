#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffles data points in two matrices same way
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
