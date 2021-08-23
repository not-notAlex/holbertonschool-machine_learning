#!/usr/bin/env python3
"""
moduel for task 2
"""

import numpy as np


def precision(confusion):
    """
    calculates sensitivity for each class in cofusion matrix
    """
    return confusion.diagonal() / np.sum(confusion, axis=0)
