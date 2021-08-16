#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """
    returns normailization constants of matrix
    """
    return X.mean(axis=0), X.std(axis=0)
