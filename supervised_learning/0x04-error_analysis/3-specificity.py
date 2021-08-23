#!/usr/bin/env python3

import numpy as np


def specificity(confusion):
    """
    calculates specificity in confusion matrix
    """
    false_negative = np.sum(confusion, axis=1) - confusion.diagonal()
    false_positive = np.sum(confusion, axis=0) - confusion.diagonal()
    true_negative = np.sum(confusion) - (
        false_positive + false_negative + confusion.diagonal())
    return true_negative / (false_positive + true_negative)
