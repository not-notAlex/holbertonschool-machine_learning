#!/usr/bin/env python3
"""
module for task 12
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on dataset
    """
    Z = scipy.cluster.hierarchy.linkage(X, method="ward")
    dendro = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    plt.show()
    return clss
