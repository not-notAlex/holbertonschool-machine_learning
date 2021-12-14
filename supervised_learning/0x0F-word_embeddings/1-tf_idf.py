#!/usr/bin/env python3
"""
module for task 1
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding
    """
    v = TfidfVectorizer(vocabulary=vocab)
    x = v.fit_transform(sentences)
    return x.toarray(), v.get_feature_names()
