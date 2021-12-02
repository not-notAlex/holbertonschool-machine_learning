#!/usr/bin/env python3
"""
module for task 8
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward prop for bidirectional RNN
    """
    Hf = []
    Hb = []
    h_prev = h_0
    h_next = h_t
    for t in range(X.shape[0]):
        h_prev = bi_cell.forward(h_prev, X[t])
        h_next = bi_cell.backward(h_next, X[X.shape[0] - 1 - t])
        Hf.append(h_prev)
        Hb.append(h_next)
    Hb = [x for x in reversed(Hb)]
    Hf = np.array(Hf)
    Hb = np.array(Hb)
    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
