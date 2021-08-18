#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates batch normalization layer for network
    """
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, name="layer", kernel_initializer=W)
    X = model(prev)
    mean, variance = tf.nn.moments(X, [0])
    return activation(
        tf.nn.batch_normalization(
            X, mean, variance, offset=tf.Variable(
                tf.constant(
                    0.0, shape=[n]), trainable=True),
            scale=tf.Variable(tf.constant(1.0, shape=[n]),
                              trainable=True), variance_epsilon=1e-8))
