#!/usr/bin/env python3
"""
module for task 0
"""

import tensorflow.keras as K


if __name__ == "__main__":
    pass


def preprocess_data(X, Y):
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p
