#!/usr/bin/env python3
"""
module for task 0
"""


def determinant(matrix):
    """
    calculates determinant of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) > 0:
        for i in matrix:
            if type(i) is not list:
                raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1 and len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    m = 1
    d = 0
    for i in range(len(matrix)):
        element = matrix[0][i]
        sub_matrix = []
        for x in range(len(matrix)):
            if x == 0:
                continue
            new_row = []
            for y in range(len(matrix)):
                if y == i:
                    continue
                new_row.append(matrix[x][y])
            sub_matrix.append(new_row)
        d += (element * m * determinant(sub_matrix))
        m *= -1
    return (d)
