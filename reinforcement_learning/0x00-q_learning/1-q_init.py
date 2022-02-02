#!/usr/bin/env python3
"""
module for task 1
"""

import gym
import numpy as np


def q_init(env):
    """
    initializes the q-table
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
