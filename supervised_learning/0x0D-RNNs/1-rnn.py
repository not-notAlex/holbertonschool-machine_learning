#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward prop for a simple RNN
    """
    T = X.shape[0]
    H = []
    Y = []
    H.append(h_0)
    h = h_0
    for i in range(T):
        h, y = rnn_cell.forward(h, X[i])
        H.append(h)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return H, Y
