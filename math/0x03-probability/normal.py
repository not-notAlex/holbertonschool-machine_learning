#!/usr/bin/env python3
"""
module for normal class
"""


class Normal:
    """
    represents a normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        sets mean and standard deviation according to data
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            devi = []
            for i in data:
                devi.append((i - self.mean)**2)
            self.stddev = ((1 / len(data)) * sum(devi))**0.5

    def z_score(self, x):
        """
        calculates the z score of a given x value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        calculates the x value of a given z score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x value
        """
        e = 2.7182818285
        p = 3.1415926536
        expo = -0.5 * ((x - self.mean) / self.stddev)**2
        return (1 / (self.stddev * (2 * p)**0.5)) * e**(expo)

    def cdf(self, x):
        """
        calculates cdf for given x value
        """
        pi = 3.1415926536
        val = (x - self.mean) / (self.stddev * (2**0.5))
        erf = val - ((val**3) / 3) + ((val**5) / 10)
        erf = erf - ((val**7) / 42) + ((val**9) / 216)
        erf = erf * (2 / (pi**0.5))
        return 0.5 * (1 + erf)
