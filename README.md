# bhc

A modern approach to the Bayesian hierarchical clustering algorithm.

Bayesian Hierarchical Clustering (BHC) is an agglomerative tree-based method for identifying underlying population structures ("clusters"). BHC was introduced by K. Heller and Z. Ghahramani as a way to approximate the more computationally intensive infinite Gaussian mixture model.

The advantage of these models over their counterparts lies in the fact that an *ex ante* number of clusters does not need to be specified. Instead, the Bayesian paradigm allows for regularized flexibility via a prior placed on the cluster concentration parameter $\alpha$. This module is build using object-oriented programing (OOP) methodologies found in python to build a module that can be used similar manner to the popular scikit-learn library. As with many Bayesian methods, the increased flexibility of BHC comes at a computational cost and the increased risk of poor results due to misspecified priors. 

---

## Algorithm Description
 
The core of the BHC algorithm relies on a Bayesian hypothesis test in which two alternatives are compared:

1) $H_1$ is the hypothesis that two clusters $D_i$ and $D_j$ were generated from the same distribution $p(x | \theta)$ with the prior distribution for $\theta$ being $p(\theta | \beta)$. The probability of clusters $i$ and $j$ being generated from the same distribution is defined as $p(D_k|H_1)$. The posterior for this hypothesis is:

$$
r_k = \frac{\pi_k p(D_k|H_1)}{\pi_k p(D_k|H_1) + (1 - \pi_k) p(D_i|T_i)p(D_j|T_j)}
$$

Note that $\pi_k$ is the prior probability of a merge occuring for clusters $i$ and $j$. This makes the denominator of this expression the Bayesian *evidence*.

2) $H_2$ is the hypothesis that the two clusters $D_i$ and $D_j$ were generated from two independent distributions and therefore should *not* be joined together as cluster $D_k$. The probability of $H_2$ is calculated as $p(D_k|H_2) = p(D_i|T_i)p(D_j|T_j)$ where $T_i$ and $T_j$ are the subclusters being examined.

All existing clusters are compared and joined based on the cluster with the highest posterior merge probability $r_k$.

---

## Development Plan

1) OOP

2) Need to create an all-encompassing structure to contain all components - think sklearn style models/objects.

3) Need to create a `cluster` object suitable for a "union-find" structure/algorithm.

4) data types:

    4.a) univariate gaussian
    
    4.b) multivariate gaussian
    
    4.c) dirchlet-multinomial

5) Pytest

6) https://www.python.org/dev/peps/pep-0008/

7) Use sphinx for documentation: https://www.sphinx-doc.org/en/master/
