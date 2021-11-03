#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np


def variance(X, C):
    """
    calculates intra-cluster variance of data set
    """
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    var = np.sum(np.min(np.sqrt(np.sum(
        (X - C[:, np.newaxis])**2, axis=-1)), axis=0) ** 2)
    return np.sum(var)
