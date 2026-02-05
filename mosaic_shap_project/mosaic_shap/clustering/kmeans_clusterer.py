import numpy as np
from sklearn.cluster import KMeans
from .base import Clusterer

class KMeansClusterer(Clusterer):
    def __init__(self, n_clusters: int = 8, random_state: int = 0, **kwargs):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)

    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(Z)
