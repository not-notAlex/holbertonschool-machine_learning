#!/usr/bin/env python3
"""
module for binomial class
"""


class Binomial:
    """
    represents a binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        sets lambtha according to data
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            d = []
            for i in data:
                d.append((i - mean)**2)
            v = sum(d) / len(d)
            t = 1 - (v / mean)
            self.n = int(round(mean / t))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        nf, kf, sf = 1, 1, 1
        for i in range(1, self.n + 1):
            nf = nf * i
        for i in range(1, k + 1):
            kf = kf * i
        for i in range(1, self.n - k + 1):
            sf = sf * i
        return (nf / (kf * sf)) * self.p**k * (1 - self.p)**(self.n - k)

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
