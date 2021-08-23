#!/usr/bin/env python3
"""
module for task 4
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates F1 score of a confusion matrix
    """
    p = precision(confusion)
    r = sensitivity(confusion)
    return (2 * p * r) / (p + r)
