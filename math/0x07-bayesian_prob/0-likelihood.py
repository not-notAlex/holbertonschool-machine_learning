#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def likelihood(x, n, P):
    """
    calculates likelihood of obtaining data
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for value in P:
        if value > 1 or value < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
    fact_coefficient = np.math.factorial(
        n) / (np.math.factorial(n - x) * np.math.factorial(x))
    return fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
