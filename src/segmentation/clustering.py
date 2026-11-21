"""Comparison of three clustering methods on the scaled RFM features."""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

SEED = 42


def fit_predict(method, X, k, seed=SEED):
    if method == "K-means":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
    if method == "CAH (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if method == "Gaussian mixture":
        return GaussianMixture(n_components=k, covariance_type="full", n_init=5,
                               random_state=seed).fit_predict(X)
    raise ValueError(method)


METHODS = ("K-means", "CAH (Ward)", "Gaussian mixture")


def scores(X, labels):
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def compare(X, ks=range(2, 9)):
    """Internal validity scores for every method and number of clusters."""
    rows = []
    for method in METHODS:
        for k in ks:
            labels = fit_predict(method, X, k)
            rows.append({"method": method, "k": k, **scores(X, labels)})
    return pd.DataFrame(rows)


def stability(X, method, k, n_runs=20, frac=0.8, seed=SEED):
    """Mean adjusted Rand index between the full-data partition and partitions
    fitted on random 80 % subsamples (1 = perfectly stable)."""
    rng = np.random.default_rng(seed)
    reference = fit_predict(method, X, k)
    aris = []
    for i in range(n_runs):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        labels = fit_predict(method, X[idx], k, seed=seed + i + 1)
        aris.append(adjusted_rand_score(reference[idx], labels))
    return float(np.mean(aris)), float(np.std(aris))
