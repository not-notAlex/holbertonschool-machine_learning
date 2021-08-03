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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        gets the weights
        """
        return self.__W

    @property
    def b(self):
        """
        gets the bias
        """
        return self.__b

    @property
    def A(self):
        """
        gets activation
        """
        return self.__A
