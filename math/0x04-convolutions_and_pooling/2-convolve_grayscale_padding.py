#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a same convolution on greyscale images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph, pw = padding
    images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), 'constant', constant_values=0)
    ch = h + (2 * ph) - kh + 1
    cw = w + (2 * pw) - kw + 1
    result = np.zeros((m, ch, cw))
    for x in range(ch):
        for y in range(cw):
            result[:, x, y] = np.sum(
                images[:, x: x + kh, y: y + kw] * kernel, axis=1).sum(axis=1)
    return result
