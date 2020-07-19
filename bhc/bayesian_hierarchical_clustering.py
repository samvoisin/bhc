"""
Bayesian Hierarchical Clustering
Author: Sam Voisin
July 2020
bayesian_hierarchical_clustering.py

Class `BHC`. This is the primary object for user interface in the bhc library.
"""

import numpy as np
from bhc.cluster import Cluster

import scipy.linalg as la
from scipy.special import gamma


def _find(clusteri: Cluster):
    """
    find parent node of cluster i

    :param clusteri: a Cluster object

    :return: label of cluster i parent node (type int)
    """
    while clusteri is not clusteri.parent:
        clusteri = clusteri.parent
    return clusteri.parent


def _union(clusteri: Cluster, clusterj: Cluster):
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
    # pass data vectors to parent cluster
    parent.points = np.r_[parent.points, child.points]


def _get_mrgnl_likelihood(dvecs, params: dict):
    """
    calculate marginal likelihood for the Normal-Inverse Wishart family of distributions

    :param dvecs: data vectors; a numpy array
    :param params: dictionary of prior parameters
    """
    N, k = dvecs.shape
    dvecsum = dvecs.sum(axis=0).reshape(1, 2)
    muvec = params["multivariate_normal"]["mean"].reshape(1, 2)
    # calculate Sprime
    Sprime = (
            params["invwishart"]["scale"] + dvecs.T @ dvecs +
            params["invwishart"]["r"] * N / (N + params["invwishart"]["r"]) * muvec.T @ muvec +
            1 / (N + params["invwishart"]["r"]) * dvecsum.T @ dvecsum -
            params["invwishart"]["r"] / (N + params["invwishart"]["r"]) *
            (muvec @ dvecsum.T + dvecsum @ muvec.T)
    )
    # calculate nu prime
    nuprime = params["invwishart"]["df"] + N
    # calculate gamma function values for marginal likelihood
    gnumer = np.array([gamma((nuprime + 1 - d) / 2) for d in range(1, k + 1)]).prod()
    gdenom = np.array([gamma((params["invwishart"]["df"] + 1 - d) / 2) for d in range(1, k + 1)]).prod()
    # calculate marginal likelihood
    mrgnl_likhd = (
            (2 * np.pi) ** (-N * k / 2) * (params["invwishart"]["r"] / (N + params["invwishart"]["r"])) ** (k / 2) *
            la.det(params["invwishart"]["scale"]) ** (params["invwishart"]["df"] / 2) *
            la.det(Sprime) ** (-nuprime / 2) *
            2 ** (N * k / 2) * (gnumer - gdenom)
    )
    return mrgnl_likhd

def _get_posterior_merge_prob(clusti: Cluster, clustj: Cluster, params: dict):
    dvecs = np.r_[clusti.points, clustj.points]
    mrgnl_likhd = _get_mrgnl_likelihood(dvecs, params)  # marginal likelihood
    nk, d = dvecs.shape
    dk = clusti.alpha * gamma(nk) + clusti.d * clustj.d
    pik = clusti.alpha * gamma(nk) / dk  # merge hypothesis prior
    treek_prob = pik * mrgnl_likhd + (1 - pik) * clusti.pmp * clustj.pmp  # tree distribution; bayes rule denominator
    return pik * mrgnl_likhd / treek_prob




class BHC:
    """
    This is the primary object for user interface in the bhc library.
    """

    def __init__(self, data: np.ndarray, alpha: float, params: dict):
        """
        initialize the bhc object with the data set and hyper-parameters.

        :param data: an `n` by `d` array of data to be clustered where n is number of data points and d is dimension
        :param alpha: cluster concentration hyper-parameter
        :param params: a dictionary of hyper-parameters for prior distributions
        """
        self.data = data
        self.n_data = data.shape[0]
        self.alpha = alpha
        self.params = params
        # n by n table for storing posterior merge probabilities
        self.pmp_table = np.zeros(shape=(self.n_data, self.n_data))
        # n by n adjacency matrix for graph representation of tree/clusters
        self.adj_mat = np.zeros(shape=(self.n_data, self.n_data)) + np.eye(self.n_data)
        self.clusters = {n: Cluster(c.reshape(1, -1), self.alpha, n) for n, c in enumerate(data)}
        # check clusters initialized with data as numpy arrays
        for v in self.clusters.values():
            assert type(v.points) == np.ndarray, "Data must be numpy array type"

    def fit(self):
        """
        Build hierarchy tree without pruning

        :return:
        """
        # calculate first round of pairwise posterior merge probabilities
        for i in self.clusters.keys():
            for j in self.clusters.keys():
                if i < j:
                    self.pmp_table[i, j] = _get_posterior_merge_prob(self.clusters[i], self.clusters[j], self.params)


if __name__ == "__main__":
    x = np.arange(24).reshape(12, 2)
    tree = BHC(x, 0.1, {"tst": 1})
    for k, v in tree.clusters.items():
        print(f"points: {v.points}")
        print(f"labels: {v.label}")

    c1 = tree.clusters.pop(0)

    for k, v in tree.clusters.items():
        print(k)
        print(v.parent)
        _union(c1, v)

    print(c1.points)
    print(tree.clusters[5].parent.label)
