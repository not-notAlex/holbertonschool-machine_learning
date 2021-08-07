#!/usr/bin/env python3
"""
module for neural-network class
"""

import numpy as np


class NeuralNetwork:
    """
    network of nodes with one hidden layer
    """
    def __init__(self, nx, nodes):
        """
        sets weights and bias of nodes in network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """
        calculates forward propagation of the network
        """
        total1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-total1))
        total2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-total2))
        return self.A1, self.A2

    def cost(self, Y, A):
        """
        calculates the cost of model using logistic regression
        """
        m = Y.shape[1]
        e = 1.0000001
        cost = (1 / m) * -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(e - A)))
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the network's predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        evaluation = np.rint(A2).astype(np.int)
        return evaluation, cost

    @property
    def W1(self):
        """
        gets W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        gets b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        gets A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        gets W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        gets b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        gets A2
        """
        return self.__A2
