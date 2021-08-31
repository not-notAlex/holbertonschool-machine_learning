#!/usr/bin/env python3
"""
module for task 12
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
