#!/usr/bin/env python3
"""
moduel for task 0
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    determines state of markov chain after iterations
    """
    prob = s
    for i in range(t):
        prob = np.matmul(result, P)
    return prob
