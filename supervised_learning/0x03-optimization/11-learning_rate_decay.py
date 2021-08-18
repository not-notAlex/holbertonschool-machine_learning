#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates learning rate using inverse time decay
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
