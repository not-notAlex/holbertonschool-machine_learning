#!/usr/bin/env python3
"""
module for exponential class
"""


class Exponential:
    """
    represents a exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        sets lambtha according to data
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        calculates the value of the pdf with given time period
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * e**(-self.lambtha * x)

    def cdf(self, x):
        """
        calculates cdf for a given time period
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - e**(-self.lambtha * x)
