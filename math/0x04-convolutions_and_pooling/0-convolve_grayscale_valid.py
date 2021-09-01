#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs convolution on greyscale images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    result = np.zeros((m, h - kh + 1, w - kw + 1))
    for x in range(h - kh + 1):
        for y in range(w - kw + 1):
            result[:, x, y] = np.sum(
                images[:, x: x + kh, y: y + kw] * kernel, axis=1).sum(axis=1)
    return result
