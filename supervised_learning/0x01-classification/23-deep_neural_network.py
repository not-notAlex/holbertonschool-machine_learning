#!/usr/bin/env python3
"""
module for Deep Neural Network class
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        performs one pass of gradient descent on the network
        """
        m = Y.shape[1]
        m = 1 / m
        prev = []
        for i in range(self.L, 0, -1):
            A = cache["A{}".format(i - 1)]
            if i == self.L:
                prev.append(cache["A{}".format(i)] - Y)
            else:
                dzp = prev[self.L - i - 1]
                Ai = cache["A{}".format(i)]
                prev.append(np.matmul(wp.T, dzp) * (Ai * (1 - Ai)))
            dW = m * np.matmul(prev[self.L - i], A.T)
            db = m * np.sum(prev[self.L - i], axis=1, keepdims=True)
            wp = self.weights["W{}".format(i)]
            self.__weights["W{}".format(i)] = (
                self.weights["W{}".format(i)] - (alpha * dW))
            self.__weights["b{}".format(i)] = (
                self.weights["b{}".format(i)] - (alpha * db))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        trains the neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        steps = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            if i % step == 0 and verbose:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))
                steps.append(cost)
            self.gradient_descent(Y, cache, alpha)
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        steps.append(cost)
        if verbose:
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph:
            x_points = np.arange(0, iterations + 1, step)
            y_points = np.asarray(steps)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

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
