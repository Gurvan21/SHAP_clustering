import numpy as np
import umap

class UMAPReducer:
    def __init__(self, n_components: int = 2, n_neighbors: int = 20, min_dist: float = 0.1, random_state: int = 0):
        self.model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(Z)

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.transform(Z)
