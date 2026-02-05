from abc import ABC, abstractmethod
import numpy as np

class Clusterer(ABC):
    @abstractmethod
    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
