import numpy as np
from sklearn.decomposition import PCA

class PCAReducer:
    def __init__(self, n_components: int = 20, random_state: int = 0):
        self.model = PCA(n_components=n_components, random_state=random_state)

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(Z)

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return self.model.transform(Z)
