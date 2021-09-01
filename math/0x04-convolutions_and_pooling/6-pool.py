#!/usr/bin/env python3
"""
module for task 6
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((h - kh) // sh) + 1
    pw = ((w - kw) // sw) + 1
    result = np.zeros((m, ph, pw, c))
    a = 0
    for x in range(0, (h - kh + 1), sh):
        b = 0
        for y in range(0, (w - kw + 1), sw):
            if mode == 'max':
                result[:, a, b, :] = np.max(images[:, x:x + kh, y:y + kw, :], axis=(1, 2))
            elif mode == 'avg':
                result[:, a, b, :] = np.average(images[:, x:x + kh, y:y + kw, :], axis=(1, 2))
            b = b + 1
        a = a + 1
    return result
