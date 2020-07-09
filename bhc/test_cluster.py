"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
test_cluster.py

test methods for Cluster class
"""

import pytest
import numpy as np
from cluster import Cluster


class TestCluster:

    def __init__(self):
        """
        initialize data needed to perform testing
        """
        self.x = np.linspace(0, 100, 20)
        self.labels = [i for i in range(5)]
        self.alpha = 1
        self.clusters = [Cluster(i, self.alpha, self.labels[n]) for n, i in enumerate(self.x)]

    @pytest.mark.xfail()
    def test_initialization(self):
        """
        test initial conditions

        :return:
        """
        pass

