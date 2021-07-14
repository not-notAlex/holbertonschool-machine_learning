#!/usr/bin/env python3
"""
module for task 7
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrixs
    """
    result = []
    try:
        if axis == 0:
            for i in mat1:
                result.append(i.copy())
            for i in mat2:
                result.append(i.copy())
            return result
        elif axis == 1:
            for x in range(0, len(mat1)):
                result.append([])
                for y in mat1[x]:
                    result[x].append(y)
            for x in range(0, len(result)):
                for y in mat2[x]:
                    result[x].append(y)
            return result
    except:
        return None
