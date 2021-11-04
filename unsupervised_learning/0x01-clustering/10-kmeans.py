#!/usr/bin/env python3
"""
module for task 10
"""

import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset
    """
    kmean = sklearn.cluster.KMeans(k)
    kmean.fit(X)
    return kmean.cluster_centers_, kmean.labels_
