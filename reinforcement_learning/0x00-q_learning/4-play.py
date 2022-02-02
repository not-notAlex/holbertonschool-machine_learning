#!/usr/bin/env python3
"""
module for task 4
"""

import gym
import numpy as np


def play(env, Q, max_steps=100):
    """
    has the trained agent play an episode
    """
    state = env.reset()
    env.render()
    for i in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    return reward
