#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    performs gradient descent on network using L2
    """
    m = Y.shape[1]
    m = 1 / m
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        l2 = (lambtha * m) * weights["W{}".format(i)]
        dw = (np.matmul(dz, cache["A{}".format(i - 1)].T) * m) + l2
        db = (np.sum(dz, axis=1, keepdims=True)) * m
        dz = np.matmul(weights["W{}".format(i)].T, dz) * (
            1 - (cache["A{}".format(i - 1)] ** 2))
        weights["W{}".format(i)] = (
            weights["W{}".format(i)] - (alpha * dw))
        weights["b{}".format(i)] = (
            weights["b{}".format(i)] - (alpha * db))
