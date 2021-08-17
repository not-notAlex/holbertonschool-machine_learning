#!/usr/bin/env python3
"""
moduel for task 5
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    returns an operation that trains the network
    """
    gd = tf.train.GradientDescentOptimizer(alpha)
    return gd.minimize(loss)
