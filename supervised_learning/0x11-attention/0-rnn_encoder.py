#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """
        initializes hidden states for RNN cell to zeros
        """
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """
        call method
        """
        return self.gru(self.embedding(x), initial_state=initial)
