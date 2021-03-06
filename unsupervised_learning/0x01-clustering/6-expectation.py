#!/usr/bin/env python3
"""
module for task 6
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates expectation step in EM for a GMM
    """
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return (None, None)
    if type(S) is not np.ndarray or type(pi) is not np.ndarray:
        return (None, None)
    if len(X.shape) != 2 or len(S.shape) != 3:
        return (None, None)
    if len(pi.shape) != 1 or len(m.shape) != 2:
        return (None, None)
    if m.shape[1] != X.shape[1]:
        return (None, None)
    if S.shape[2] != S.shape[1]:
        return (None, None)
    if S.shape[0] != pi.shape[0] or S.shape[0] != m.shape[0]:
        return (None, None)
    if np.min(pi) < 0:
        return (None, None)
    g = np.zeros([pi.shape[0], X.shape[0]])
    for i in range(pi.shape[0]):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    return g / g.sum(axis=0), np.sum(np.log(g.sum(axis=0)))
