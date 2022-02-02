#!/usr/bin/env python3
"""
module for task 2
"""

import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    uses epsilon-greedy to determine next action
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return action
