#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np
import tensorflow as tf


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    performs gradient descent on network using L2
    """
    m = Y.shape[1]
    m = 1 / m
    prev = []
    for i in range(L, 0, -1):
        A = cache["A{}".format(i - 1)]
        if i == L:
            prev.append(cache["A{}".format(i)] - Y)
        else:
            dzp = prev[L - i - 1]
            Ai = cache["A{}".format(i)]
            prev.append(np.matmul(wp.T, dzp) * (Ai * (1 - Ai)))
        dW = m * np.matmul(prev[L - i], A.T)
        db = m * np.sum(prev[L - i], axis=1, keepdims=True)
        wp = weights["W{}".format(i)]
        weights["W{}".format(i)] = (
            weights["W{}".format(i)] - (alpha * dW))
        weights["b{}".format(i)] = (
            weights["b{}".format(i)] - (alpha * db))
