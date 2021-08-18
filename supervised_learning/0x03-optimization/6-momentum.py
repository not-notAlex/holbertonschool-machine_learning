#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates training op using gradient descent w/ momentum algorithm
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
