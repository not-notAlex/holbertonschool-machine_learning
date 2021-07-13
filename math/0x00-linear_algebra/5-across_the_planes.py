#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for x in range(0, len(mat1)):
        result.append([])
        for y in range(0, len(mat1[0])):
            result[x].append(mat1[x][y] + mat2[x][y])
    return result
