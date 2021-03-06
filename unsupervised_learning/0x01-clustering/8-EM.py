#!/usr/bin/env python3
"""
module for task 8
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    performs expectation maximum for a GMM
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) is not int or type(iterations) is not int:
        return (None, None, None, None, None)
    if k <= 0 or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) is not float or tol < 0:
        return (None, None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None, None)
    pi, m, S = initialize(X, k)
    g, ll = expectation(X, pi, m, S)
    ll_old = 0
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, ll.round(5)))
        pi, m, S = maximization(X, g)
        g, ll = expectation(X, pi, m, S)
        if np.abs(ll_old - ll) <= tol:
            break
        ll_old = ll
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1, ll.round(5)))
    return pi, m, S, g, ll
