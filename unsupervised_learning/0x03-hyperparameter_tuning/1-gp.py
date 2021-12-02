#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


class GaussianProcess:
    """
    Gaussian Process class
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        sqdist = np.sum(X_init**2, 1).reshape(-1, 1) + np.sum(
            X_init**2, 1) - 2 * np.dot(X_init, X_init.T)
        self.K = sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

    def kernel(self, X1, X2):
        """
        calculates covariance matrix
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(
            X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        predicts mean standard deviation
        """
        ks = self.kernel(self.X, X_s)
        kss = self.kernel(X_s, X_s)
        kinv = np.linalg.inv(self.K)
        mu = np.dot(ks.T, kinv).dot(self.Y).reshape((X_s.shape[0]))
        sigma = np.diag(kss - np.dot(ks.T, kinv).dot(ks))
        return mu, sigma