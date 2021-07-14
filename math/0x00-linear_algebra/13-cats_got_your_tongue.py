#!/usr/bin/env python3
"""
module for task 13
"""


def np_cat(mat1, mat2, axis=0):
    """
    concantenates two numpy matrix along axis
    """
    import numpy as np
    return np.concatenate((mat1, mat2), axis)
