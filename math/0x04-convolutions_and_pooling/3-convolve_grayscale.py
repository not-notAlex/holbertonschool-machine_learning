#!/usr/bin/env python3
"""
module for task 3
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on greyscale images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    else:
        ph, pw = padding
    images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), 'constant', constant_values=0)
    ch = ((h + (2 * ph) - kh) // sh) + 1
    cw = ((w + (2 * pw) - kw) // sw) + 1
    result = np.zeros((m, ch, cw))
    a = 0
    for x in range(0, (h + (2 * ph) - kh + 1), sh):
        b = 0
        for y in range(0, (w + (2 * pw) - kw + 1), sw):
            result[:, a, b] = np.sum(
                images[:, x: x + kh, y: y + kw] * kernel, axis=1).sum(axis=1)
            b = b + 1
        a = a + 1
    return result
