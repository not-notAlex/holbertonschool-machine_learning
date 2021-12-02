#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np


class GRUCell:
    """
    class for GRU Cell
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        performs softmax function
        """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """
        performs sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """
        performs one step of forward prop
        """
        U = np.hstack((h_prev, x_t))
        z = self.sigmoid(np.dot(U, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(U, self.Wr) + self.br)
        U = np.hstack((h_prev * r, x_t))
        h_next = np.multiply(z, np.tanh(np.dot(
            U, self.Wh) + self.bh)) + np.multiply((1 - z), h_prev)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
