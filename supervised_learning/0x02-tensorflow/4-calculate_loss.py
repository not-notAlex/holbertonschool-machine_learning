#!/usr/bin/env python3
"""
module for task 4
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    calculates loss of prediction
    """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred,)
