#!/usr/bin/env python3
"""
module for task 4
"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5,
                   seed=0, workers=1):
    """
    creates and trains a gensim fastText model
    """
    model = FastText(size=size, window=window, min_count=min_count,
                     workers=workers, sg=not cbow,
                     seed=seed, negative=negative)
    model.build_vocab(sentences=sentences)
    model.train(
        sentences=sentences, total_examples=len(sentences), epochs=epochs)
    return model
