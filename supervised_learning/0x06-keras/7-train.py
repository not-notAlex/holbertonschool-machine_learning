#!/usr/bin/env python3
"""
module for task 7
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    """
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        def learning_rate(epoch):
            """
            performs the learning rate
            """
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(K.callbacks.LearningRateScheduler(learning_rate, 1))
    return network.fit(
        x=data, y=labels, batch_size=batch_size,
        epochs=epochs, validation_data=validation_data,
        callbacks=callbacks, verbose=verbose, shuffle=shuffle)
