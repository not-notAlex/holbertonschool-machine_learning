#!/usr/bin/env python3
"""
module for task 3
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    back propagation on pooling network
    """
    m, h, w, c = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((m, h_prev, w_prev, c))
    for i in range(m):
        for ki in range(c):
            for x in range(h):
                for y in range(w):
                    a = x * sh
                    z = y * sw
                    if mode is 'max':
                        mask = np.where(
                            A_prev[i, a: a + kh, z: z + kw, ki] == np.max(
                                A_prev[i, a: a + kh, z: z + kw, ki]), 1, 0)
                    elif mode is 'avg':
                        mask = np.ones((kh, kw)) / (kh * kw)
                    dA_prev[i, a: a + kh, z: z + kw, ki] += (
                        mask * dA[i, x, y, ki])
    return dA_prev
