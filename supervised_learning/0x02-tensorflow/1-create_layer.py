#!/usr/bin/env python3
"""
module for task 1
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    returns the tensor output of the layer
    """
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            name="layer", kernel_initializer=W)
    return layer(prev)
