#!/usr/bin/env python3
"""
module for task 6
"""


def cat_arrays(arr1, arr2):
    """
    contatenates two arrays
    """
    result = []
    for i in arr1:
        result.append(i)
    for i in arr2:
        result.append(i)
    return result
