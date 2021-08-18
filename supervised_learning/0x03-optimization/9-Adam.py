#!/usr/bin/env python3
"""
module for task 0
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates variables using Adam algorithm
    """
    vdW = (beta1 * v) + ((1 - beta1) * grad)
    sdW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    vdW_change = vdW / (1 - (beta1 ** t))
    sdW_change = sdW / (1 - (beta2 ** t))
    var -= alpha * (vdW_change / (epsilon + (sdW_change ** (1 / 2))))
    return var, vdW, sdW
