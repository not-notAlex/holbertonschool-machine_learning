#!/usr/bin/env python3
"""
moduel for task 6
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs forward algorithm on HMM
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros([N, T])
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for x in range(1, T):
        for y in range(N):
            F[y, x] = F[:, x - 1].dot(
                Transition[:, y]) * Emission[y, Observation[x]]
    return F


def backward(Observation, Emission, Transition, Initial):
    """
    performs backward algorithm on HMM
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros([N, T])
    B[:, T - 1] = np.ones((N))
    for x in range(T - 2, -1, -1):
        for y in range(N):
            B[y, x] = (B[:, x + 1] * Emission[:, Observation[x + 1]]).dot(
                Transition[y, :])
    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs Baum-Welch algorithm on HMM
    """
    T = Observations.shape[0]
    M, N = Emission.shape
    for n in range(1, iterations):
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((M, M, T - 1))
        for x in range(T - 1):
            den = np.dot(np.dot(F[:, x].T, Transition) *
                         Emission[:, Observations[x + 1]].T, B[:, x + 1])
            for y in range(M):
                num = (F[y, x] * Transition[y] *
                       Emission[:, Observations[x + 1]].T * B[:, x + 1].T)
                xi[y, :, x] = num / den
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((
            gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        denom = np.sum(gamma, axis=1)
        for i in range(N):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)
        Emission = np.divide(Emission, denom.reshape((-1, 1)))
    return Transition, Emission
