#!/usr/bin/env python3
"""
module for task 4
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    """
    d = {}
    d["A0"] = X
    for i in range(L):
        w = weights["W{}".format(i + 1)]
        b = weights["b{}".format(i + 1)]
        z = np.matmul(w, d["A{}".format(i)]) + b
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        if i == L - 1:
            A = np.exp(z)
            A = A / np.sum(A, axis=0, keepdims=True)
        else:
            A = np.tanh(z)
            A *= dropout
            A /= keep_prob
            d["D{}".format(i + 1)] = dropout
        d["A{}".format(i + 1)] = A
    return d
