#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on greyscale images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    pw = kw // 2
    ph = kh // 2
    images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), 'constant', constant_values=0)
    result = np.zeros((m, h, w))
    for x in range(h - kh + 1):
        for y in range(w - kw + 1):
            result[:, x, y] = np.sum(
                images[:, x: x + kh, y: y + kw] * kernel, axis=1).sum(axis=1)
    return result
