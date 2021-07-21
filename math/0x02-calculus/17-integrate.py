#!/usr/bin/env python3
"""
module for task 17
"""


def poly_integral(poly, C=0):
    """
    calculates integral of polynomial
    """
    result = []
    if type(poly) is not list or type(C) is not int or len(poly) == 0:
        return None
    if len(poly) > 0:
        result.append(C)
    for i in range(0, len(poly)):
        if i == 0:
            result.append(poly[i])
        else:
            num = poly[i] / (i + 1)
            if float.is_integer(num):
                num = int(num)
            result.append(num)
    if len(result) == 2 and (result[0] == 0 or result[1] == 0) and C != 0:
        result.remove(0)
    return result
