#!/usr/bin/env python3
"""
module for task 3
"""

import gym
import numpy as np
epsilonGreedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs q-learning
    """
    rewards = []
    max_eps = epsilon
    for i in range(episodes):
        state = env.reset()
        current_reward = 0
        for j in range(max_steps):
            action = epsilonGreedy(Q, state, epsilon)
            n_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[n_state, :]) - Q[state, action])
            state = n_state
            current_reward += reward
            if done:
                break
        epsilon = min_epsilon + (
            max_eps - min_epsilon) * np.exp(-epsilon_decay * i)
        rewards.append(current_reward)
    return Q, rewards
