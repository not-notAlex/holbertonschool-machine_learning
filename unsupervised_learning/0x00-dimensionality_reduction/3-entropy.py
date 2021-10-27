#!/usr/bin/env python3
"""
module for task 3
"""

import numpy as np


def HP(Di, beta):
    """
    calculates Shannon entropy
    """
    Pi = np.exp(-Di * beta) / np.sum(np.exp(-Di * beta))
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
