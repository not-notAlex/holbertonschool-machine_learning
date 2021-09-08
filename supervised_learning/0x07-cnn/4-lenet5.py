#!/usr/bin/env python3
"""
module for task 4
"""

import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5
    """
    ki = tf.contrib.layers.variance_scaling_initializer()
    L1 = tf.layers.Conv2D(6, kernel_size=(
        5, 5), padding='same', activation=tf.nn.relu, kernel_initializer=ki)(x)
    L2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(L1)
    L3 = tf.layers.Conv2D(16, kernel_size=(
        5, 5), padding='valid', activation=tf.nn.relu,
                          kernel_initializer=ki)(L2)
    L4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(L3)
    out = tf.layers.Flatten()(L4)
    L5 = tf.layers.Dense(
        120, activation=tf.nn.relu, kernel_initializer=ki)(out)
    L6 = tf.layers.Dense(84, activation=tf.nn.relu, kernel_initializer=ki)(L5)
    L7 = tf.layers.Dense(10, kernel_initializer=ki)(L6)
    out = tf.nn.softmax(L7)
    loss = tf.losses.softmax_cross_entropy(y, logits=L7)
    train = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.math.argmax(L7, axis=1)
    y_out = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))
    return out, train, loss, accuracy
