#!/usr/bin/env python3
"""
module for task 6
"""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    """
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    return network.fit(
        x=data, y=labels, batch_size=batch_size,
        epochs=epochs, validation_data=validation_data,
        callbacks=callbacks, verbose=verbose, shuffle=shuffle)
