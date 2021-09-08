#!/usr/bin/env python3
"""
module for task 4
"""

import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5
    """
    weights_initializer = tf.contrib.layers.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, kernel_initializer=weights_initializer)(x)
    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C1)
    C3 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu, kernel_initializer=weights_initializer)(P2)
    P4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C3)
    out = tf.layers.Flatten()(P4)
    F5 = tf.layers.Dense(120, activation=tf.nn.relu, kernel_initializer=weights_initializer)(out)
    F6 = tf.layers.Dense(84, activation=tf.nn.relu, kernel_initializer=weights_initializer)(F5)
    F7 = tf.layers.Dense(10, kernel_initializer=weights_initializer)(F6)
    softmax = tf.nn.softmax(F7)
    loss = tf.losses.softmax_cross_entropy(y, logits=F7)
    op = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.math.argmax(F7, axis=1)
    y_out = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))
    return softmax, op, loss, accuracy
