#!/usr/bin/env python3
"""
module for task 7
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        return True, count
    else:
        return False, count
