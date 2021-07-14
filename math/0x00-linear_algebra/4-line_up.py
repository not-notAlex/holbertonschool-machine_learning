#!/usr/bin/env python3
"""
module for task 4
"""


def add_arrays(arr1, arr2):
    """
    adds two array element-wise
    """
    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(0, len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
