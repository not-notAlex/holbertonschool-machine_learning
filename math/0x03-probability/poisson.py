#!/usr/bin/env python3
"""
module for poisson class
"""


class Poisson:
    """
    represents a poisson distribution
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        factorial = 1
        e = 2.7182818285
        for i in range(1, k + 1):
            factorial = factorial * i
        return (e**-self.lambtha * self.lambtha**k) / factorial

    def cdf(self, k):
        """
        calculates cdf for given number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        probability = 0
        for i in range(k + 1):
            probability = probability + self.pmf(i)
        return probability
