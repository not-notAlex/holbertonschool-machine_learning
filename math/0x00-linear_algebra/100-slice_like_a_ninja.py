#!/usr/bin/env python3
"""
module for task 15
"""


def np_slice(matrix, axes={}):
    """
    slices matrix along a specific axis
    """
    slices = len(matrix.shape) * [slice(None)]
    for k, v in axes.items():
        slices[k] = slice(*v)
    return matrix[tuple(slices)]
