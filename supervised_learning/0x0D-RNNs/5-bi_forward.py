#!/usr/bin/env python3
"""
module for task 5
"""

import numpy as np


class BidirectionalCell:
    """
    class for Bidirectional Cell
    """
    def __init__(self, i, h, o):
        """
        class constructor
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward prop one step
        """
        return np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Whf) + self.bhf)
