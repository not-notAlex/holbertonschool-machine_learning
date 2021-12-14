#!/usr/bin/env python3
"""
module for task 2
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    creates and trains a gensim word2vec
    """
    model = Word2Vec(sentences=sentences, size=size, window=window,
                     min_count=min_count, workers=workers, sg=not cbow,
                     seed=seed, negative=negative)
    model.train(
        sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
