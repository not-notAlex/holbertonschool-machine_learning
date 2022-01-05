#!/usr/bin/env python3
"""
module for task 1
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    self attention class
    """
    def __init__(self, units):
        """
        class constructor
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        call method
        """
        s_prev = tf.expand_dims(s_prev, 1)
        e = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        c = tf.nn.softmax(e, axis=1) * hidden_states
        c = tf.reduce_sum(c, axis=1)
        return c, tf.nn.softmax(e, axis=1)
