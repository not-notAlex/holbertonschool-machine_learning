#!/usr/bin/env python3
"""
moduel for task 8
"""


def mat_mul(mat1, mat2):
    """
    performs matrix multiplication
    """
    columns = len(mat1[0])
    rows = len(mat2)
    if columns != rows:
        return None
    result = [[]]
    for i in range(1, len(mat1)):
        result.append([])
    for x in result:
        for y in range(0, len(mat2[0])):
            x.append(0)
    for x in range(0, len(mat1)):
        for y in range(0, len(mat1[0])):
            for z in range(0, len(mat2[0])):
                result[x][z] = result[x][z] + (mat1[x][y] * mat2[y][z])
    return result
