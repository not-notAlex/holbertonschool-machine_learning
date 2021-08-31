#!/usr/bin/env python3
"""
module for task 2
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization
    """
    network.compile(optimizer=K.optimizers.Adam(
        lr=alpha, beta_1=beta1, beta_2=beta2), loss='', metrics=['accuracy'])
    return None
