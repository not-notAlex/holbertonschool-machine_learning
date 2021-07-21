#!/usr/bin/env python3
"""
module for task 17
"""


def poly_integral(poly, C=0):
    """
    calculates integral of polynomial
    """
    result = []
    result.append(C)
    for i in range(0, len(poly)):
        if i == 0:
            result.append(poly[i])
        else:
            num = poly[i] / (i + 1)
            if float.is_integer(num):
                num = int(num)
            result.append(num)
    return result
