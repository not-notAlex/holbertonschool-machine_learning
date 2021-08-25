#!/usr/bin/env python3
"""
module for task 6
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer using Dropout
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.layers.Dropout(keep_prob)
    result = tf.layers.Dense(
        units=n, activation=activation, name="layer",
        kernel_initializer=w, kernel_regularizer=l2)
    return result(prev)
