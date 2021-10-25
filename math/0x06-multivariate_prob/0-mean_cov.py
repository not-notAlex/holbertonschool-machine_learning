#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def mean_cov(X):
    """
    calculates mean and covariance of data
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0)
    cov = np.dot(X.T, X - mean) / (n - 1)
    return mean, cov
