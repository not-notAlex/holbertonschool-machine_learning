#!/usr/bin/env python3
"""
module for task 3
"""

import numpy as np


class LSTMCell:
    """
    class for LSTM Cell
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        performs one step of forward prop
        """
        U = np.hstack((h_prev, x_t))
        u = self.sigmoid(np.dot(U, self.Wu) + self.bu)
        c_bar = np.tanh(np.dot(U, self.Wc) + self.bc)
        c_next = self.sigmoid(np.dot(
            U, self.Wf) + self.bf) * c_prev + u * c_bar
        o = self.sigmoid(np.dot(U, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, c_next, y
