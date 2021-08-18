#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains model using min-batches
    """
    m = X_train.shape[0]
    with tf.Session() as session:
        save = tf.train.import_meta_graph(load_path + '.meta')
        save.restore(session, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for i in range(epochs + 1):
            tcost = session.run(loss, feed_dict={x: X_train, y: Y_train})
            taccu = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
            vcost = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            vaccu = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(tcost))
            print("\tTraining Accuracy: {}".format(taccu))
            print("\tValidation Cost: {}".format(vcost))
            print("\tValidation Accuracy: {}".format(vaccu))
            if i == epochs:
                break
            x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)
            if m % batch_size == 0:
                mini = m // batch_size
            else:
                mini = m // batch_size + 1
            for step in range(mini):
                lo = step * batch_size
                hi = (step + 1) * batch_size
                if hi > m:
                    hi = m
                session.run(train_op, feed_dict={x: x_shuffle[lo:hi, :],
                                                 y: y_shuffle[lo:hi, :]})
                if (step + 1) % 100 == 0:
                    cost = session.run(
                        loss, feed_dict={x: x_shuffle[lo:hi, :],
                                         y: y_shuffle[lo:hi, :]})
                    accu = session.run(
                        accuracy, feed_dict={x: x_shuffle[lo:hi, :],
                                             y: y_shuffle[lo:hi, :]})
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(accu))
        return save.save(session, save_path)
