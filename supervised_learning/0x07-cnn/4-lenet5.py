#!/usr/bin/env python3
"""
module for task 4
"""

import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5
    """
    initializer = (tf.contrib.layers.variance_scaling_initializer())
    out = tf.layers.Conv2D(6, 5, padding='same', activation='relu', kernel_initializer=initializer)(x)
    out = tf.layers.MaxPooling2D(2, 2)(out)
    out = tf.layers.Conv2D(16, 5, padding='valid', activation='relu', kernel_initializer=initializer)(out)
    out = tf.layers.MaxPooling2D(2, 2)(out)
    out = tf.layers.Flatten()(out)
    out = tf.layers.Dense(120, activation='relu', kernel_initializer=initializer)(out)
    out = tf.layers.Dense(84, activation='relu', kernel_initializer=initializer)(out)
    out = tf.layers.Dense(10, activation='softmax', kernel_initializer=initializer)(out)
    loss = tf.losses.softmax_cross_entropy(y, out)
    train = tf.train.AdamOptimizer().minimize(loss)
    equal = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    return out, train, loss, accuracy
