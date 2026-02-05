import numpy as np
from sklearn.cluster import AgglomerativeClustering
from .base import Clusterer

class AgglomerativeClusterer(Clusterer):
    def __init__(self, n_clusters: int = 8, linkage: str = "ward", **kwargs):
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)

    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(Z)
