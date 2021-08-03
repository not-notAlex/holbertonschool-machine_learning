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

    def forward_prop(self, X):
        """
        calculates forward propagation of the neuron
        """
        total = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-total))
        return self.A

    def cost(self, Y, A):
        """
        calculates the cost of model using regression
        """
        m = Y.shape[1]
        e = 1.0000001
        cost = (1 / m) * -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(e - A)))
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the neurons predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        evaluation = np.rint(A).astype(np.int)
        return evaluation, cost

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
