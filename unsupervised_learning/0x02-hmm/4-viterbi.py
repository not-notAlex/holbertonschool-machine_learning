#!/usr/bin/env python3
"""
moduel for task 4
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculates sequence for hidden states in HMM
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    D = np.zeros([N, T])
    path = np.zeros(T)
    phi = np.zeros([N, T])
    D[:, 0] = Initial.T * Emission[:, Observation[0]]
    for x in range(1, T):
        for y in range(N):
            D[y, x] = np.max(
                D[:, x - 1] * Transition[:, y]) * Emission[y, Observation[x]]
            phi[y, x] = np.argmax(D[:, x - 1] * Transition[:, y])
    path[T - 1] = np.argmax(D[:, T - 1])
    for i in range(T - 2, -1, -1):
        path[i] = phi[int(path[i + 1]), i + 1]
    P = np.max(D[:, T - 1:], axis=0)[0]
    path = [int(i) for i in path]
    return path, P
