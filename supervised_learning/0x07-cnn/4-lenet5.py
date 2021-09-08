#!/usr/bin/env python3
"""
module for task 4
"""

import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5
    """
    ki = (tf.contrib.layers.variance_scaling_initializer())
    L1 = tf.layers.Conv2D(6, 5, padding='same', activation='relu', kernel_initializer=ki)(x)
    L2 = tf.layers.MaxPooling2D(2, 2)(L1)
    L3 = tf.layers.Conv2D(16, 5, padding='valid', activation='relu', kernel_initializer=ki)(L2)
    L4 = tf.layers.MaxPooling2D(2, 2)(L3)
    L5 = tf.layers.Flatten()(L4)
    L6 = tf.layers.Dense(120, activation='relu', kernel_initializer=ki)(L5)
    L7 = tf.layers.Dense(84, activation='relu', kernel_initializer=ki)(L6)
    out = tf.layers.Dense(10, activation='softmax', kernel_initializer=ki)(L7)
    loss = tf.losses.softmax_cross_entropy(y, out)
    train = tf.train.AdamOptimizer().minimize(loss)
    max_pred = tf.argmax(out, 1)
    equal = tf.equal(tf.argmax(y, 1), max_pred)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    return out, train, loss, accuracy
