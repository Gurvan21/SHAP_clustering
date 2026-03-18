from __future__ import annotations
from typing import Any, Sequence, List, Optional
import numpy as np
import shap
from ..base import Explainer

def _predict_fn(model: Any):
    # same idea as before: output must be 1D (n,)
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]
    return lambda X: model.predict(X)

class Order1OwenSHAP(Explainer):
    """
    Owen values via SHAP Partition masker.
    groups: list of groups, each group is a list of feature indices.
    Returns (n_samples, n_features).
    """
    def __init__(
        self,
        background: np.ndarray,
        groups: Sequence[Sequence[int]],
        max_evals: int = 2000,
        output: str = "auto",          # "auto" usually fine; you can expose it if needed
    ):
        self.background = np.asarray(background)
        self.groups = [list(g) for g in groups]
        self.max_evals = int(max_evals)
        self.output = output

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X)

        # Partition masker implements Owen-style hierarchical Shapley over groups.
        # API shap>=0.51: l'argument s'appelle `clustering` (anciennement `groups`).
        masker = shap.maskers.Partition(self.background, clustering=self.groups)

        # Use shap.Explainer with masker; wrap model into a predictable 1D function
        f = _predict_fn(model)
        expl = shap.Explainer(f, masker, output_names=None)

        exp = expl(X, max_evals=self.max_evals)

        vals = exp.values
        if isinstance(vals, list):
            vals = vals[1]
        vals = np.asarray(vals)

        # Safety for (n,p,k) outputs
        if vals.ndim == 3:
            k = 1 if vals.shape[2] > 1 else 0
            vals = vals[:, :, k]

        return vals
