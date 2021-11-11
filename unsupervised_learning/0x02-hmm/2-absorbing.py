#!/usr/bin/env python3
"""
moduel for task 2
"""

import numpy as np


def absorbing(P):
    """
    determines of markov chain is absorbing
    """
    ab = np.where(np.diag(P) == 1)
    B = np.delete(np.delete(np.copy(P), ab[0], 0), ab[0], 1)
    In = np.identity(B.shape[0])
    try:
        result = np.linalg.inv(In - B)
        return True
    except Exception:
        return False
