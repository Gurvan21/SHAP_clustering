"""Single reduction interface: method='pca' | 'umap'."""
from __future__ import annotations

import numpy as np


class Reducer:
    """Thin wrapper: fit_transform(Z) and transform(Z)."""

    def __init__(self, model):
        self.model = model

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(Z)

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.transform(Z)


def create_reducer(
    method: str,
    *,
    n_components: int = 2,
    random_state: int | None = 0,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    **kwargs,
) -> Reducer:
    """Build a Reducer. method='pca' or 'umap'."""
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components, random_state=random_state, **kwargs)
    elif method == "umap":
        import umap
        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")
    return Reducer(model)
