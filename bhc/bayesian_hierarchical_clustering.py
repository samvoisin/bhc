"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
bayesian_hierarchical_clustering.py

Class `BHC`. This is the primary object for user interface in the bhc library.
"""

import numpy as np
from cluster import Cluster


def _find(clusteri: Cluster):
    """
    find parent node of cluster i

    :param clusteri: a Cluster object

    :return: label of cluster i parent node (type int)
    """
    while clusteri is not clusteri.parent:
        clusteri = clusteri.parent
    return clusteri.parent


def _union(self, clusteri: Cluster, clusterj: Cluster):
    """
    Joint the pair of clusters with highest posterior merge probability

    :param clusteri: Cluster object to be merged with clusterj
    :param clusterj: Cluster object to be merged with clusteri
    """
    if clusteri.rank >= clusterj.rank:
        parent = clusteri
        child = clusterj
    else:
        parent = clusterj
        child = clusteri
    child.parent = _find(parent)
    # increase rank of top node
    _find(parent).rank += 1







class BHC:
    """
    This is the primary object for user interface in the bhc library.
    """

    def __init__(self, data: np.ndarray, alpha: float, prior_params: dict, family="normal_in_gamma"):
        """
        Initialize the bhc object with the data set and hyper-parameters.

        :param data: an `n` by `d` array of data to be clustered where n is number of data points and d is dimension
        :param alpha: cluster concentration hyper-parameter
        :param prior_params: a dictionary of hyper-parameters for prior distributions
        :param family: distribution family to be used; this must match the prior parameter dictionary provided
        """
        self.data = data
        self.n_data, self.n_dims = data.shape
        self.alpha = alpha
        self.prior_params = prior_params
        self.family = family
        # n by n table for storing posterior merge probabilities
        self.posterior_table = np.zeros(shape=(self.n_data, self.n_dims))
        self.uf_array = np.arange(self.n_data)  # union-find tracking array
        self.clusters = {n: Cluster(c, self.alpha, n) for n, c in enumerate(data)}  # init points as individual clusters


if __name__ == "__main__":
    x = np.arange(24).reshape(12, 2)
    tree = BHC(x, 0.1, {"tst": 1})
    for k, v in tree.clusters.items():
        print(f"points: {v.points}")
        print(f"labels: {v.label}")
