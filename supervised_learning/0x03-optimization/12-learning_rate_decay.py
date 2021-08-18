#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates learning rate decay op using inverse time decay
    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
