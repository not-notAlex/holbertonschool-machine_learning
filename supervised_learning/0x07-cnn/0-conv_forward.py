#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    forward propagation on convolutional network
    """
    m, h, w, c = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    else:
        return
    images = np.pad(A_prev, ((0, 0), (
        ph, ph), (pw, pw), (0, 0)), 'constant', constant_values=0)
    ch = ((h + (2 * ph) - kh) // sh) + 1
    cw = ((w + (2 * pw) - kw) // sw) + 1
    result = np.zeros((m, ch, cw, nc))
    for i in range(nc):
        ki = W[:, :, :, i]
        a = 0
        for x in range(0, (h + (2 * ph) - kh + 1), sh):
            b = 0
            for y in range(0, (w + (2 * pw) - kw + 1), sw):
                result[:, a, b, i] = np.sum(
                    images[:, x: x + kh, y: y + kw,
                           :] * ki, axis=1).sum(axis=1).sum(axis=1)
                b = b + 1
            a = a + 1
    return result
