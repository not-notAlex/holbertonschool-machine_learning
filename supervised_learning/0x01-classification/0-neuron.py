#!/usr/bin/env python3
"""
module for neuron class
"""

import numpy as np


class Neuron:
    """
    node of a single neuron
    """
    def __init__(self, nx):
        """
        sets weights, bias, and activation of node
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
