#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset
    """
    u, s, v = np.linalg.svd(X)
    a = np.cumsum(s)
    dim = []
    l = s.shape[0]
    for i in range(l):
        if ((a[i]) / a[-1]) >= var:
            dim.append(i)
    r = dim[0] + 1
    return v.T[:, :r]
