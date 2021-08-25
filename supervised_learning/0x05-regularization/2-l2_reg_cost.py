#!/usr/bin/env python3
"""
module for task 2
"""

import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """
    claculates cost and returns tensor on L2 network
    """
    return tf.losses.get_regularization_losses() + cost
