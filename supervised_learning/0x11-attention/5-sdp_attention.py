#!/usr/bin/env python3
"""
module for task 5
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates scaled dot product attention
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        logits += (mask * -1e9)
    weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(weights, V), weights
