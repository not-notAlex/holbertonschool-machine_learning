#!/usr/bin/env python3
"""
module for task 2
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN decoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        call method
        """
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context, 1), self.embedding(x)], -1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        return self.F(output), state
