#!/usr/bin/env python3
"""
module for task 10
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves weights from network
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    loads weights into network
    """
    network.load_weights(filename)
    return None
