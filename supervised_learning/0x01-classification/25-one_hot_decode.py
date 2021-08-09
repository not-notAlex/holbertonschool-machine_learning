#!/usr/bin/env python3
"""
module for one hot function
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    decodes one-hot into label vector
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return one_hot.T.argmax(axis=1)
