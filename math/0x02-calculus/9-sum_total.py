#!/usr/bin/env python3
"""
module for task 9
"""


def summation_i_squared(n):
    """
    calculates a sum of i^2
    """
    if n < 1:
        return None
    total = int((n * (n + 1) * ((2 * n) + 1)) / 6)
    return total
