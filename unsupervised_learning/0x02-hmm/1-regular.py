#!/usr/bin/env python3
"""
moduel for task 1
"""

import numpy as np


def regular(P):
    """
    steady state probabilities of markov chain
    """
    jstat = np.argmin(abs(np.linalg.eig(P.T)[0] - 1.0))
    stat = np.linalg.eig(P.T)[1][:, jstat].real
    stat = stat / stat.sum()
    if np.min(stat) <= 0 or np.sum(stat) != 1:
        return None
    return stat[np.newaxis, :]
