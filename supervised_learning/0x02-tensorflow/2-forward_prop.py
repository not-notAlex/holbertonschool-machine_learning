#!/usr/bin/env python3

import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates forward propagation graph for network
    """
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i == 0:
            result = create_layer(x, layer_sizes[i], activations[i])
        else:
            result = create_layer(result, layer_sizes[i], activations[i])
    return result
