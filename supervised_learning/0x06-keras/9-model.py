#!/usr/bin/env python3
"""
module for task 9
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    saves a model
    """
    network.save(filename)


def load_model(filename):
    """
    loads a model
    """
    return K.models.load_model(filename)
