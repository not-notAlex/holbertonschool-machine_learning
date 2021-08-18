#!/usr/bin/env python3
"""
module for task 0
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates variable using RMSProp algorithm
    """
    dw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var -= alpha * (grad / (epsilon + (dw ** (1 / 2))))
    return var, dw
