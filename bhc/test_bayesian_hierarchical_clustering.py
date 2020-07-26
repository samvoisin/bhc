"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
test_cluster.py

test methods for Cluster class
"""


import pytest
import numpy as np
from scipy.stats import multivariate_normal
from bhc.bayesian_hierarchical_clustering import BHC

class TestBHCHelpers:
    """
    set of tests for bhc helper functions
    """

    def setup(self):
        """
        initialize data required to perform testing
        """
        pass



class TestBHC:
    """
    set of tests for BHC class
    """

    def setup(self):
        """
        initialize data required to perform testing
        """
        np.random.seed(1)
        self.test_data = multivariate_normal.rvs(mean=np.zeros(3), cov=np.eye(3), size=10)
        self.alpha = 1
        self.params1 = {}  # to be included; params1 is parameterization for mvt normal case
        self.params2 = {}  # to be included; params2 is parameterization for univariate extension case
        self.params3 = {}  # to be included; params3 is parameterization for dirichlet multinomial case

    @pytest.xfail()
    def test_bhc_init(self):
        """
        test BHC class instantiation

        :return:
        """
        pass