#!/usr/bin/env python3
"""
module for task 4
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    bayesian optimization class
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        class constructor
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.minimize = minimize
        self.xsi = xsi
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """
        calculates next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='warn'):
            if self.minimize:
                musopt = np.min(self.gp.Y)
                imp = (musopt - mu - self.xsi).reshape(-1, 1)
            else:
                musopt = np.amax(self.gp.Y)
                imp = (mu - musopt - self.xsi).reshape(-1, 1)
            ei = imp * norm.cdf(imp / sigma) + sigma * norm.pdf(imp / sigma)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei.reshape(-1)
