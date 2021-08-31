#!/usr/bin/env python3
"""
module for task 13
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using network
    """
    return network.predict(x=data, verbose=verbose)
