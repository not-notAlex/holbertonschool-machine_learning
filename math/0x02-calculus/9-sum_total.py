#!/usr/bin/env python3
"""
module for task 9
"""


def summation_i_squared(n):
    """
    calculates a sum of i^2
    """
    total = 0
    if n < 1:
        return None
    for i in range(1, n + 1):
        total = total + (i ** 2)
    return total
