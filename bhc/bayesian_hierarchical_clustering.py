"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
bayesian_hierarchical_clustering.py

Class `BHC`. This is the primary object for user interface in the bhc library.
"""

import numpy as np
from scipy import linalg as la


class BHC:
    """
    This is the primary object for user interface in the bhc library.
    """

    def __init__(self, x: np.ndarray, alpha: float, prior_params: dict, family="univariate_gaussian"):
        """
        initialize the bhc algorithm with the data set and hyper-parameters

        :param x: an `n` by `d` array of data to be clustered where n is number of data points and d is dimension
        :param alpha: cluster concentration hyper-parameter
        :param prior_params: a dictionary of hyper-parameters for prior distributions
        :param family: distribution family to be used; this must match the prior parameter dictionary provided
        """
        self.x = x
        self.data_dims = x.shape
        self.alpha = alpha
        self.prior_params = prior_params
        self.family = family
        # n by n table for storing posterior merge probabilities
        self.post_table = np.zeros(shape=(self.data_dims[0], self.data_dims[0]))

