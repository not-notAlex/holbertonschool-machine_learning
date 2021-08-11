#!/usr/bin/env python3

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    returns an operation that trains the network
    """
    gd = tf.train.GradientDescentOptimizer(alpha)
    return gd.minimize(loss)
