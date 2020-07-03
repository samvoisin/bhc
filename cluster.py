"""
Bayesian Hierarchical Clustering
Author: Sam Voisin

Cluster class
"""


class Cluster:
    """
    Cluster object.

    This object represents a cluster of two or more
    leaves (i.e. data points). Each leaf is initialized in its own cluster.

    x - a float or array input. Each instance is initialized with
    a single vector. Points are joined and clusters merge as dendrogram
    develops.
    """

    def __init__(self, x: float, label: int):
        self.points = [x]  # Cluster object initialized with a single data point
        self.label = label

    def merge cluster(self, cj: Cluster):
        """
        Merge another cluster, cj, into this cluster instance.

        cj - another cluster object to be merged.
        """
        pass
