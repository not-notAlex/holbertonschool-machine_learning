#!/usr/bin/env python3
"""
module for task 11
"""

import sklearn.mixture


def gmm(X, k):
    """
    calculates GMM from a dataset
    """
    Gmm = sklearn.mixture.GaussianMixture(k)
    params = Gmm.fit(X)
    return (params.weights_, params.means_,
            params.covariances_, Gmm.predict(X), Gmm.bic(X))
