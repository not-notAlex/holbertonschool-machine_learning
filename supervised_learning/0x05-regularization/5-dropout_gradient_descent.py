#!/usr/bin/env python3
"""
module for task 5
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    performs gradient descent on network using dropout
    """
    m = Y.shape[1]
    m = 1 / m
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (np.matmul(dz, cache["A{}".format(i - 1)].T) * m)
        db = (np.sum(dz, axis=1, keepdims=True)) * m
        if i - 1 > 0:
            dz = np.matmul(weights["W{}".format(i)].T, dz) * (
                1 - (cache["A{}".format(i - 1)] ** 2)) * (
                    cache["D{}".format(i - 1)] / keep_prob)
        weights["W{}".format(i)] = (
            weights["W{}".format(i)] - (alpha * dw))
        weights["b{}".format(i)] = (
            weights["b{}".format(i)] - (alpha * db))
