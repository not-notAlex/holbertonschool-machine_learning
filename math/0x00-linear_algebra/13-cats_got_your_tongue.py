#!/usr/bin/env python3
"""
module for task 13
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    concantenates two numpy matrix along axis
    """
    return np.concatenate((mat1, mat2), axis)
