#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset
    """
    mean = np.mean(X, axis=0, keepdims=True)
    a = X - mean
    u, s, v = np.linalg.svd(a)
    w = v.T[:, :ndim]
    return np.matmul(a, w)
