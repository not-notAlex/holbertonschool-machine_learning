#!/usr/bin/env python3
def matrix_shape(matrix):
    result = []
    try:
        size = matrix
        while True:
            result.append(len(size))
            size = size[0]
    except TypeError:
        pass
    return result
