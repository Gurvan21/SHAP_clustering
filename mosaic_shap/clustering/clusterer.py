"""Single clustering interface: wrap any estimator with fit_predict; factory for common methods."""
from __future__ import annotations

import numpy as np


class Clusterer:
    """Thin wrapper around any estimator that has fit_predict(Z) -> labels."""

    def __init__(self, model):
        self.model = model

    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(Z)


def create_clusterer(
    method: str,
    *,
    # KMeans
    n_clusters: int = 8,
    random_state: int | None = 0,
    # Agglomerative
    linkage: str = "ward",
    # HDBSCAN
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    **kwargs,
) -> Clusterer:
    """Build a Clusterer for the given method. Pass method-specific args as keyword arguments."""
    method = method.lower()
    if method == "kmeans":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
    elif method == "hdbscan":
        import hdbscan
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples if min_samples is not None else min_cluster_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}. Use one of: kmeans, agglomerative, hdbscan")
    return Clusterer(model)
