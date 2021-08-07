#!/usr/bin/env python3
"""
module for Deep Neural Network class
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Neural network with multiple hidden layers
    """
    def __init__(self, nx, layers):
        """
        sets layers, cache, and weights
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.__weights = {}
        prev_layer = nx
        for i, n in enumerate(layers, 1):
            if type(n) is not int or n < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i)] = (
                np.random.randn(n, prev_layer) * np.sqrt(2 / prev_layer))
            self.__weights["b{}".format(i)] = np.zeros((n, 1))
            prev_layer = n
        self.__L = len(layers)
        self.__cache = {}

    def forward_prop(self, X):
        """
        calculates forward propagation on network
        """
        for i in range(self.L + 1):
            if i == 0:
                self.__cache["A0"] = X
            else:
                W = self.weights["W{}".format(i)]
                b = self.weights["b{}".format(i)]
                value = np.matmul(W, self.cache["A{}".format(i - 1)]) + b
                A = 1 / (1 + np.exp(-value))
                self.__cache["A{}".format(i)] = A
        return A, self.cache

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        e = 1.0000001
        cost = (1 / m) * -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(e - A)))
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the network's predictions
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        evaluation = np.rint(A).astype(np.int)
        return evaluation, cost

    @property
    def L(self):
        """
        gets L value
        """
        return self.__L

    @property
    def cache(self):
        """
        gets cache value
        """
        return self.__cache

    @property
    def weights(self):
        """
        gets weights value
        """
        return self.__weights
