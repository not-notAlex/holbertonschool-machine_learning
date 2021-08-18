#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes unactivated output of network using batch normalization
    """
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    return gamma * ((Z - mean) / ((var + epsilon) ** 0.5)) + beta
