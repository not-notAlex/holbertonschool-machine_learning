#!/usr/bin/env python3
"""
module for task 5
"""

import numpy as np


def definiteness(matrix):
    """
    calculates definiteness of matrix
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix.T, matrix):
        return None
    ev, _ = np.linalg.eig(matrix)
    if all(ev < 0):
        return "Negative definite"
    if all(ev <= 0):
        return "Negative semi-definite"
    if all(ev > 0):
        return "Positive definite"
    if all(ev >= 0):
        return "Positive semi-definite"
    else:
        return 'Indefinite'
