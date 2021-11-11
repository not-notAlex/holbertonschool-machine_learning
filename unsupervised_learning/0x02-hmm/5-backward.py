#!/usr/bin/env python3
"""
moduel for task 5
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    performs backward algorithm for HMM
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros([N, T])
    B[:, T - 1] = np.ones((N))
    for x in range(T - 2, -1, -1):
        for y in range(N):
            B[y, x] = (B[:, x + 1] * Emission[:, Observation[x + 1]]).dot(
                Transition[y, :])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
