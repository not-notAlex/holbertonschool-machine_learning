#!/usr/bin/env python3
"""
module for task 5
"""

import numpy as np


def Q_affinities(Y):
    """
    calculates the Q affinities
    """
    n = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(
        n, np.sum(np.square(Y), 1)).T, np.sum(np.square(Y), 1)))
    num[range(Y.shape[0]), range(Y.shape[0])] = 0
    Q = num / np.sum(num)
    return (Q, num)
