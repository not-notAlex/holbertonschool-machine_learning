#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    performs K-means on dataset
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    d = X.shape[1]
    C = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        new_C = np.copy(C)
        for c in range(k):
            if c not in clss:
                new_C[c] = np.random.uniform(
                    np.amin(X, axis=0), np.amax(X, axis=0))
            else:
                new_C[c] = np.mean(X[clss == c], axis=0)
        if (new_C == C).all():
            return (C, clss)
        else:
            C = new_C
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return (C, clss)
