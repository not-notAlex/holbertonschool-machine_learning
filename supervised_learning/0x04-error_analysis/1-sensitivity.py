#!/usr/bin/env python3

import numpy as np


def sensitivity(confusion):
    """
    calculates sensitivity for each class in cofusion matrix
    """
    return confusion.diagonal() / np.sum(confusion, axis=1)
