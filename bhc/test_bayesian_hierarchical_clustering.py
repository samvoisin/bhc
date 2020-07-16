"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
test_cluster.py

test methods for Cluster class
"""


import pytest
import numpy as np
from bhc.bayesian_hierarchical_clustering import BHC

class TestBHC:
    """
    Class of tests for BHC
    """

    def setup(self):
        """
        initialize data needed to perform testing
        """
        pass

    @pytest.xfail()
    def test_bhc_init(self):
        """
        test cluster instantiation

        :return:
        """
        pass