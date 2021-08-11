#!/usr/bin/env python3

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of prediction
    """
    y = tf.math.argmax(y, axis=1)
    y_pred = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
