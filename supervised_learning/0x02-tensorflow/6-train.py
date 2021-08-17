#!/usr/bin/env python3
"""
module for task 6
"""

import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network
    """
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    create_train_op = __import__('5-create_train_op').create_train_op
    forward_prop = __import__('2-forward_prop').forward_prop
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    save = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(iterations):
            tcost = session.run(loss, feed_dict={x: X_train, y: Y_train})
            taccu = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
            vcost = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            vaccu = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(tcost))
                print("\tTraining Accuracy: {}".format(taccu))
                print("\tValidation Cost: {}".format(vcost))
                print("\tValidation Accuracy: {}".format(vaccu))
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
        tcost = session.run(loss, feed_dict={x: X_train, y: Y_train})
        taccu = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
        vcost = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
        vaccu = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        print("After {} iterations:".format(iterations))
        print("\tTraining Cost: {}".format(tcost))
        print("\tTraining Accuracy: {}".format(taccu))
        print("\tValidation Cost: {}".format(vcost))
        print("\tValidation Accuracy: {}".format(vaccu))
        return save.save(session, save_path)
