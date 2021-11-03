#!/usr/bin/env python3
"""
module for task 3
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for optimum number of clusters by variance
    """
    if type(X) is not np.ndarray:
        return (None, None)
    if type(kmin) is not int:
        return (None, None)
    if kmax is not None and type(kmax) is not int:
        return (None, None)
    if kmax is None:
        kmax = X.shape[0]
    if len(X.shape) != 2 or kmin < 1:
        return (None, None)
    if kmax is not None and kmax <= kmin:
        return (None, None)
    if type(iterations) is not int:
        return (None, None)
    if iterations <= 0:
        return (None, None)
    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k)
        results.append((C, clss))
        variances.append(variance(X, C))
    d_vars = []
    for i in range(len(variances)):
        d_vars.append(variances[0] - variances[i])
    return (results, d_vars)
