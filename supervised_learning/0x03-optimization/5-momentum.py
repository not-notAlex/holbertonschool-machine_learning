#!/usr/bin/env python3
"""
module for task 0
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates variable w/ gradient descent w/ momentum algorithm
    """
    dw = (beta1 * v) + ((1 - beta1) * grad)
    var -= alpha * dw
    return var, dw
