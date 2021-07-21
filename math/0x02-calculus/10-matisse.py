#!/usr/bin/env python3
"""
module for task 10
"""


def poly_derivative(poly):
    """
    calculates derivative of polynomial
    """
    result = []
    if type(poly) is not list or poly is None:
        return None
    for i in range(1, len(poly)):
        result.append(i * poly[i])
    if len(result) == 1:
        result.append(0)
    if len(result) == 0:
        return None
    return result
