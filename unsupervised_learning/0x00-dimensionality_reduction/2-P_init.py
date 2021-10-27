#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np


def P_init(X, perplexity):
    """
    initializes variables required to calculate t-SNE
    """
    n = X.shape[0]
    m = np.matmul(X, -X.T)
    D = np.add(np.add(-2 * np.dot(X, X.T), np.sum(np.square(X), 1)).T, np.sum(np.square(X), 1))
    P = np.zeros((X.shape[0], X.shape[0]))
    betas = np.ones((X.shape[0], 1))
    H = np.log2(perplexity)
    return D, P, betas, H
