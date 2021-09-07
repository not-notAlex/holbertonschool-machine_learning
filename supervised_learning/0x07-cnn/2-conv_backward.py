#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    back propagation on convolutional network
    """
    m, h, w, c = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    else:
        return
    images = np.pad(A_prev, ((0, 0), (
        ph, ph), (pw, pw), (0, 0)), 'constant', constant_values=0)
    dA = np.zeros((m, h_prev + (2 * ph), w_prev + (2 * pw), c_prev))
    dW = np.zeros((kh, kw, c_prev, c))
    for i in range(m):
        for ki in range(c):
            for x in range(h):
                for y in range(w):
                    a = x * sh
                    z = y * sw
                    dA[i, a: a + kh, z: z + kw, :] += (
                        dZ[i, x, y, ki] * W[:, :, :, ki])
                    dW[:, :, :, ki] += (
                        images[i, a: a + kh, z: z + kw, :] * dZ[i, x, y, ki])
    if padding is 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dW, db
