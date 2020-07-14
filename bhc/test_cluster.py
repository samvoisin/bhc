"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
test_cluster.py

test methods for Cluster class
"""


import pytest
import numpy as np
from bhc.cluster import Cluster


class TestCluster:
    """
    Class of tests for Cluster
    """

    def setup(self):
        """
        initialize data needed to perform testing
        """
        x = np.linspace(0, 100, 20)
        self.alpha = 1
        self.clusters = [Cluster(x=i, alpha=self.alpha, label=n) for n, i in enumerate(x)]

    def test_cluster_init(self):
        """
        test cluster instantiation

        :return:
        """
        assert np.array([c.alpha for c in self.clusters]).sum() == self.alpha * len(self.clusters)
