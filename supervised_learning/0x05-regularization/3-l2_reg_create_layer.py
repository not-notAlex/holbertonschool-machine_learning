#!/usr/bin/env python3
"""
module for task 3
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tf layer with L2 regularization
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation, name="layer", kernel_initializer=w, kernel_regularizer=l2)
    return layer(prev)
