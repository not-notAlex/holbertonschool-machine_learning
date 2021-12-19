#!/usr/bin/env python3
"""
module for task 1
"""

import numpy as np


def counter(phrase, n):
    """
    returns dict with count of words
    """
    tuple_list = []
    for x in range(len(phrase) - n + 1):
        tuple_list.append(tuple(i for i in phrase[x:x + n]))
    dict = {}
    for x in range(len(phrase) - n + 1):
        key = tuple(i for i in phrase[x:x + n])
        if key not in dict:
            dict[key] = tuple_list.count(key)
    return dict


def count_clip(references, sentence, n):
    """
    counts clip
    """
    res = {}
    ct_sentence = counter(sentence, n)
    for ref in references:
        ct_ref = counter(ref, n)
        for k in ct_ref:
            if k in res:
                res[k] = max(ct_ref[k], res[k])
            else:
                res[k] = ct_ref[k]
    count_clip = {k: min(ct_sentence.get(k, 0), res.get(
        k, 0)) for k in ct_sentence}
    return count_clip


def modified_precision(references, sentence, n):
    """
    modified precision
    """
    ct_clip = count_clip(references, sentence, n)
    return sum(ct_clip.values()) / float(max(sum(
        counter(sentence, n).values()), 1))


def ngram_bleu(references, sentence, n):
    """
    calculates n-gram BLEU score for sentence
    """
    W = [0.25 for x in range(4)]
    Pn = [modified_precision(
        references, sentence, n) for ngram, _ in enumerate(W, start=1)]
    closest_ref_idx = np.argmin([abs(len(
        x) - len(sentence)) for x in references])
    r = len(references[closest_ref_idx])
    BP = np.exp(1 - (float(r) / len(sentence)))
    if len(sentence) > r:
        BP = 1
    score = np.sum([(wn * np.log(Pn[i])) if Pn[i] != 0 else 0
                    for i, wn in enumerate(W)])
    return BP * np.exp(score)
