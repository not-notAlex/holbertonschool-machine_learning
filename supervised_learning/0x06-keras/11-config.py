#!/usr/bin/env python3
"""
module for task 11
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    saves network in JSON
    """
    output = network.to_json()
    with open(filename, 'w+') as f:
        f.write(output)
    return None


def load_config(filename):
    """
    loads network from JSON
    """
    with open(filename, 'r') as f:
        inputs = f.read()
    return K.models.model_from_json(inputs)
