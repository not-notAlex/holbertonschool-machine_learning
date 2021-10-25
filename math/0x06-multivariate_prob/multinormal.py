#!/usr/bin/env python3
"""
module for task 2 and 3
"""

import numpy as np


class MultiNormal:
    """
    Multinormal class
    """

    def __init__(self, data):
        """
        class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.cov = np.matmul(
            data - self.mean, data.T - self.mean.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        calculates the PDF at a data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        d = self.cov.shape[0]
        if x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        mult = np.matmul(np.matmul(
            (x - self.mean).T, np.linalg.inv(self.cov)), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        return pdf[0][0]
