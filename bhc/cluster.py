"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
cluster.py

Class `Cluster` representing a cluster of data points. This the fundamental component of the union-find structure.
"""

import numpy as np


def _calc_merge_posterior(clusti, clustj):
    """
    Calculate the posterior probability of merging with another cluster

    :param clusti: Cluster class instance
    :param clustj: Cluster class instance
    """
    pass


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
        self.points = [x]  # Cluster object initialized with a single data point
        self.label = label  # cluster label
        self.parent = self.label  # parent node of cluster initialized as self
        self.rank = 1  # tier of hierarchy tree
        self.alpha = alpha
        self.merge_prior = 1  # prior probability of merging clusters
        # d parameter controls merge hypothesis prior using information in subtrees; it represents the observation that
        # subtrees which were not merged on previous iterations may not share a distribution
        self.d = alpha
        # posterior merge probability; updates value to the posterior merge probability when it becomes a child node
        self.pmp = 1

    def pull_vectors_up(self):
        """
        Pull data vectors from child nodes for use in posterior calculations

        :return:
        """
        pass


if __name__ == "__main__":
    pass
