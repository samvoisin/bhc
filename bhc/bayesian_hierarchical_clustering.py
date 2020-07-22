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
    join a pair of clusters; child node will always merge with highest rank node in cluster

    :param clusteri: Cluster object to be merged with clusterj
    :param clusterj: Cluster object to be merged with clusteri
    :return: tuple containing two Cluster objects (parent, child)
    """
    if _find(clusteri).rank >= _find(clusteri).rank:
        parent = _find(clusteri)
        child = clusterj
    else:
        parent = _find(clusterj)
        child = clusteri
    child.parent = _find(parent)  # point child node to highest ranking node in parent cluster
    parent.rank += 1  # increase rank of highest ranking node in new cluster
    # pass data vectors to parent cluster
    parent.points = np.r_[parent.points, child.points]
    return parent, child


def _get_table_coordinates(table: np.ndarray, fcn: callable):
    """
    get coordinates from of max value from `pmp_table`; these should correspond to cluster labels

    :param table: either `pmp_table` or `adj_mat`
    :param fcn: a function returning a scalar value from `table` (e.g. max or min)
    :return: tuple of integers
    """
    i, j = np.where(table == fcn(table))
    i = i[0]
    j = j[0]
    return i, j


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
    """
    calculate posterior merge probability for clusters i and j given prior params `params`

    :param clusti: cluster i
    :param clustj: cluster j
    :param params: dictionary of prior parameters
    :return: posterior probability for the merge hypothesis; type float
    """
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
        # n by n adjacency matrix for directed graph representation of tree; directed edges go from rows to columns
        self.adj_mat = np.zeros(shape=(self.n_data, self.n_data)) + np.eye(self.n_data)
        self.clusters = {n: Cluster(c.reshape(1, -1), self.alpha, n) for n, c in enumerate(data)}
        # check clusters initialized with data as numpy arrays
        for v in self.clusters.values():
            assert type(v.points) == np.ndarray, "Data must be numpy array"

    def fit(self):
        """
        Build hierarchy tree without pruning

        :return:
        """
        # calculate first round of pairwise posterior merge probabilities
        for m in self.clusters.keys():
            for n in self.clusters.keys():
                if m < n:
                    self.pmp_table[m, n] = _get_posterior_merge_prob(self.clusters[m], self.clusters[n], self.params)
        # return coordinates of max posterior merge probability; these should correspond to cluster labels
        i_label, j_label = _get_table_coordinates(self.pmp_table, np.max)
        parent, child = _union(self.clusters[i_label], self.clusters[j_label])  # merge clusters i and j
        # update graph with an edge from child node to parent node
        self.adj_mat[child.label, parent.label] = self.pmp_table.max()
        # recalculate posterior merge probabilities for new cluster only
        child_labels = []
        candidate_pmp = np.zeros(self.n_data)
        for candidate in self.clusters.values():
            if candidate.parent is parent:  # if parent already contains candidate cluster, append and move on
                child_labels.append(candidate.label)
            else:
                candidate_pmp[candidate.label] = _get_posterior_merge_prob(parent, candidate, self.params)
        # update posterior merge probability table
        for c in child_labels:
            self.pmp_table[c, :] = candidate_pmp


if __name__ == "__main__":
    x = np.arange(24).reshape(12, 2)
    tree = BHC(x, 0.1, {"tst": 1})
    tree.fit()
