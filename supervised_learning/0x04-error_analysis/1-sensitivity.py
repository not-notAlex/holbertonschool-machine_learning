#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def sensitivity(confusion):
    """
    calculates sensitivity for each class in cofusion matrix
    """
    return confusion.diagonal() / np.sum(confusion, axis=1)
