#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


class RNNCell:
    """
    class for RNN Cell
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        performs softmax function
        """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        performs forward propogation
        """
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Wh) + self.bh)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
