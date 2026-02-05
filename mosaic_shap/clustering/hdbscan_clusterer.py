import numpy as np
import hdbscan
from .base import Clusterer

class HDBSCANClusterer(Clusterer):
    def __init__(self, min_cluster_size: int = 50, min_samples: int | None = None, **kwargs):
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples if min_samples is not None else min_cluster_size,
            **kwargs
        )

    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(Z)
