#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def normalize(X, m, s):
    """
    normalizes a matrix
    """
    return (X - m) / s
