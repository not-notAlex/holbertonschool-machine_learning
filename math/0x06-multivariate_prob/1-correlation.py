#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def correlation(C):
    """
    calculates a correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    D = np.sqrt(np.diag(C))
    inverse = 1 / np.outer(D, D)
    return inverse * C
