#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates cost with L2 regularization
    """
    lw = 0
    for i in range(1, L + 1):
        lw = lw + np.linalg.norm(weights['W' + str(i)])**2
    return (cost + ((lambtha / (2 * m)) * lw))
