from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Explainer(ABC):
    @abstractmethod
    def compute(self, model: Any, X: np.ndarray, **kwargs):
        raise NotImplementedError

def predict_score(model: Any, X: np.ndarray) -> np.ndarray:
    """1D score used for model-agnostic explainers.
    - predict_proba[:,1] -> logit
    - decision_function
    - predict
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s).reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)
