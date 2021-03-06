"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
cluster.py

Class `Cluster` representing a cluster of data points. This the fundamental component of the union-find structure.
"""

import numpy as np
from scipy.special import gamma


class Cluster:
    """
    Object representing a cluster of data points. This the fundamental element of the union-find structure.
    """

    def __init__(self, x: float or np.ndarray, alpha: float, label: int):
        """
        Initialize cluster for a single data point. On initialization each cluster is a singleton set (i.e. a leaf).

        :param x: a single data vector. This may be a float value or an array of floats.
        :param alpha: cluster concentration parameter. This should be common to all clusters at initialization.
        :param label: a unique integer identifier for the cluster.
        """
        self.label = label  # cluster label
        self.parent = self  # parent node of cluster initialized as self
        self.points = x  # Cluster object initialized with a single data point
        self.rank = 1  # tier of hierarchy tree
        self.alpha = alpha
        self.merge_prior = 1.0  # prior probability of merging clusters
        # d parameter controls merge hypothesis prior using information in subtrees; it represents the observation that
        # subtrees which were not merged on previous iterations may not share a distribution
        self.d = alpha
        self.clust_marg_prob = 1.0  # marginal probability of data in this cluster; this is p(Dk|Tk) in original paper

    def update_d_param(self, clustj):
        """
        update `d` parameter after merging with `clustj`. `d` parameter is used for calculating merge prior

        :params clustj: Cluster being merged into this Cluster instance
        """
        self.d = self.alpha * np.nan_to_num(gamma(self.points.shape[0])) + self.d*clustj.d


if __name__ == "__main__":
    pass
