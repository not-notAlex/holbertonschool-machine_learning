#!/usr/bin/env python3
"""
module for task 3
"""


def matrix_transpose(matrix):
    """
    returns transposed matrix
    """
    result = []
    for x in range(0, len(matrix[0])):
        result.append([])
        for y in matrix:
            result[x].append(y[x])
    return result
