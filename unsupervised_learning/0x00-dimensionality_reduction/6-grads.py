#!/usr/bin/env python3
"""
module for task 6
"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    calculates the gradients of Y
    """
    Q, num = Q_affinities(Y)
    diff = P - Q
    dY = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        dY[i, :] = np.sum(np.tile(
            diff[:, i] * num[:, i], (Y.shape[1], 1)).T * (Y[i, :] - Y), 0)
    return dY, Q
