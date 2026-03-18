from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Explainer(ABC):
    @abstractmethod
    def compute(self, model: Any, X: np.ndarray, **kwargs):
        raise NotImplementedError


def predict_score(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Helper utilisé par certains estimateurs d'interactions (ordre 2).
    Garantit un retour 1D float, en essayant predict_proba puis predict.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # binaire: on prend la proba de la classe positive
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba.squeeze(), dtype=float)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float).squeeze()
    raise TypeError("Model must implement predict or predict_proba for predict_score.")
