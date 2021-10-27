#!/usr/bin/env python3
"""
module for task 7
"""

import numpy as np


def cost(P, Q):
    """
    calculates the cost of t-SNE transformation
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    return np.sum(P * np.log(P / Q))
