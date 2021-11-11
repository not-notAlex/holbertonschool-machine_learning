#!/usr/bin/env python3
"""
moduel for task 3
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs forward algorithm for HMM
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros([N, T])
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for x in range(1, T):
        for y in range(N):
            F[y, x] = F[:, x - 1].dot(
                Transition[:, y]) * Emission[y, Observation[x]]
    P = np.sum(F[:, T - 1:], axis=0)[0]
    return P, F
