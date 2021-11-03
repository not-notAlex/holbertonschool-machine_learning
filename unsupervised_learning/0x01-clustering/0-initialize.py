#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def initialize(X, k):
    """
    initializes centroids for K-Means
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    d = X.shape[1]
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(
        np.min(X, axis=0), np.max(X, axis=0), size=(k, d))
    return centroids
