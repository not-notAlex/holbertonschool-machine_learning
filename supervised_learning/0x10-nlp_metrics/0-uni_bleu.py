#!/usr/bin/env python3
"""
module for task 0
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    calculates unigram BLEU score for sentence
    """
    sentence_length = len(sentence)
    references_length = []
    words = {}
    for translation in references:
        references_length.append(len(translation))
        for word in translation:
            if word in sentence and word not in words.keys():
                words[word] = 1
    total = sum(words.values())
    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    BLEU = np.exp(1 - float(len(references[index])) / float(sentence_length))
    if sentence_length > len(references[index]):
        BLEU = 1
    return BLEU * np.exp(np.log(total / sentence_length))
