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


def _get_merge_prior(clusti: Cluster, clustj: Cluster):
    """
    calculate prior probability of merging cluster i and cluster j

    :param clusti: one of two candidate clusters
    :param clustj: one of two candidate clusters
    :return: scalar float value in interval (0,1)
    """
    ni = clusti.points.shape[0]
    nj = clusti.points.shape[0]
    nk = ni+nj
    dk = clusti.alpha * gamma(nk) + clusti.d * clustj.d
    pik = clusti.alpha * gamma(nk) / dk
    return pik


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
    nk, p = dvecs.shape
    pik = _get_merge_prior(clusti, clustj)  # merge hypothesis prior
    # treek_prob is tree distribution; bayes rule denominator
    treek_prob = pik * mrgnl_likhd + (1 - pik) * clusti.clust_marg_prob * clustj.clust_marg_prob
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
        self.current_clusters = list(self.clusters.keys())
        # check clusters initialized with data as numpy arrays
        for v in self.clusters.values():
            assert type(v.points) == np.ndarray, "Data must be numpy array"

    def update_tables(self, parent: Cluster, child: Cluster):
        """
        update posterior merge probability table and adjacency matrix after child cluster merges into parent

        :param parent: cluster being merged into
        :param child: cluster being merged into parent
        """
        self.adj_mat[child.label, parent.label] = self.pmp_table.max()  # update graph w/ edge from child to parent
        self.pmp_table[child.label, :] = np.zeros(self.n_data)  # zero all posterior merge probabilities for child
        self.pmp_table[:, child.label] = np.zeros(self.n_data)  # zero all posterior merge probabilities for child
        self.current_clusters.remove(child.label)  # remove child label from list of current clusters

    def fit(self):
        """
        Build hierarchy tree without pruning
        """
        # calculate pairwise posterior merge probability table
        for m in self.clusters.keys():
            for n in self.clusters.keys():
                if m < n:
                    self.pmp_table[m, n] = _get_posterior_merge_prob(self.clusters[m], self.clusters[n], self.params)
        self.pmp_table = self.pmp_table + self.pmp_table.T
        while len(self.current_clusters) > 1:  # agglomeration loop
            # return coordinates of max posterior merge probability; these should correspond to cluster labels
            i_label, j_label = _get_table_coordinates(self.pmp_table, np.max)
            parent, child = _union(self.clusters[i_label], self.clusters[j_label])  # merge clusters i and j
            parent.update_d_param(child)  # update parent prior merge probability (pik)
            # update marginal probability of data in treek; this is p(Dk|Tk) in original paper where k is parent node
            parent.merge_prior = _get_merge_prior(parent, child)
            parent.clust_marg_prob = (
                    parent.merge_prior * _get_mrgnl_likelihood(parent.points, self.params) +
                    (1-parent.merge_prior) * parent.clust_marg_prob * child.clust_marg_prob
            )
            self.update_tables(parent, child)
            # recalculate posterior merge probabilities for new cluster only
            candidate_pmp = np.zeros(self.n_data)
            for candidate_label in self.current_clusters:
                if candidate_label == parent.label:
                    continue  # skip this iteration's parent node
                else:
                    candidate_pmp[candidate_label] = _get_posterior_merge_prob(parent,
                                                                               self.clusters[candidate_label],
                                                                               self.params)
            self.pmp_table[parent.label, :] = candidate_pmp
            self.pmp_table[:, parent.label] = candidate_pmp


if __name__ == "__main__":
    from scipy.stats import multivariate_normal
    # generate trial data set
    mu1 = np.array([3, 4])
    Sigma1 = np.array([0.1, 0, 0, 0.1]).reshape(2, 2)

    mu2 = np.ones(2) * -1
    Sigma2 = np.array([0.8, 0.3, 0.3, 0.8]).reshape(2, 2)

    mvn1 = multivariate_normal(mean=mu1, cov=Sigma1)
    mvn2 = multivariate_normal(mean=mu2, cov=Sigma2)

    np.random.seed(100)
    n1 = 100
    n2 = 150
    x1 = mvn1.rvs(n1)
    x2 = mvn2.rvs(n2)
    x = np.r_[x1, x2]
    labs = np.r_[np.zeros(n1), np.ones(n2)]

    params = {
        "multivariate_normal": {"mean": x.mean(axis=0), "cov": x.std()},
        "invwishart": {"df": 10, "scale": np.eye(2), "r": 1}  # r is a scaling factor on the prior precision of the mean
    }

    tree = BHC(data=x, alpha=1, params=params)
    tree.fit()
