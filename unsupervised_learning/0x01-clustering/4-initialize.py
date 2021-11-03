#!/usr/bin/env python3
"""
module for task 4
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for Gaussian Mixture model
    """
    if type(X) is not np.ndarray or type(k) is not int:
        return (None, None, None)
    if len(X.shape) != 2 or k <= 0:
        return (None, None, None)
    d = X.shape[1]
    m, _ = kmeans(X, k)
    S = np.zeros((k, d, d))
    S[:] = np.identity(d)
    pi = np.zeros((k))
    pi[:] = 1 / k
    return pi, m, S
