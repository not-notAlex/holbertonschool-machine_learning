#!/usr/bin/env python3
"""
module for task 0
"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embed matrix
    """
    v = CountVectorizer(vocabulary=vocab)
    x = v.fit_transform(sentences)
    return x.toarray(), v.get_feature_names()
