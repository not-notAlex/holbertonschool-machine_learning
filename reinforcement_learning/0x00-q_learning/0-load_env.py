#!/usr/bin/env python3
"""
module for task 0
"""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    loads FrozenLakeEnv from gym
    """
    return gym.make(
        "FrozenLake-v0", map_name=map_name, desc=desc, is_slippery=is_slippery)
