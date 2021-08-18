#!/usr/bin/env python3
"""
module for task 0
"""


def moving_average(data, beta):
    """
    calculates weighted moving average
    """
    ema = []
    v = 0
    for i in range(len(data)):
        v = (v * beta) + ((1 - beta) * data[i])
        ema.append(v / (1 - (beta ** (i + 1))))
    return ema
