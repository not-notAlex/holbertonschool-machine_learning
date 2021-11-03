#!/usr/bin/env python3
"""
module for task 5
"""

import numpy as np


def pdf(X, m, S):
    """
    calculates PDF of Gaussian distribution
    """
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return None
    if type(S) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(S.shape) != 2:
        return None
    if len(m.shape) != 1:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None
    N = np.sqrt((2*np.pi)**d * np.linalg.det(S))
    fac = np.einsum('...k,kl,...l->...', X-m, np.linalg.inv(S), X-m)
    P = np.exp(-fac / 2) / N
    P[np.where(P < 1e-300)] = 1e-300
    return P
