"""Order-1 SHAP: single class with method='tree' | 'permutation' | 'kernel'."""
from __future__ import annotations

from typing import Any

import numpy as np
import shap

from ..base import Explainer


def _normalize_shap_values(vals) -> np.ndarray:
    """Handle list (multi-class) and 3D output from shap."""
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        k = 1 if vals.shape[2] > 1 else 0
        vals = vals[:, :, k]
    return vals


class Order1Explainer(Explainer):
    """
    Order-1 SHAP values (feature attributions).
    method: "tree" (TreeExplainer), "permutation" (PermutationExplainer), "kernel" (KernelExplainer).
    """

    def __init__(
        self,
        method: str = "tree",
        *,
        # tree
        check_additivity: bool = False,
        feature_perturbation: str = "auto",
        model_output: str = "raw",
        # permutation / kernel
        background: np.ndarray | None = None,
        max_evals: int = 2000,
        nsamples: int = 200,
        **kwargs,
    ):
        self.method = method.lower()
        if self.method not in ("tree", "permutation", "kernel"):
            raise ValueError('method must be one of: "tree", "permutation", "kernel"')
        self.check_additivity = check_additivity
        self.feature_perturbation = feature_perturbation
        self.model_output = model_output
        self.background = np.asarray(background) if background is not None else None
        self.max_evals = max_evals
        self.nsamples = nsamples
        self._kwargs = kwargs

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X)
        if self.method == "tree":
            expl = shap.TreeExplainer(
                model,
                feature_perturbation=self.feature_perturbation,
                model_output=self.model_output,
            )
            vals = expl.shap_values(X, check_additivity=self.check_additivity)
            return _normalize_shap_values(vals)
        if self.method == "permutation":
            if self.background is None:
                raise ValueError("method='permutation' requires background")
            expl = shap.Explainer(model, self.background, algorithm="permutation")
            exp = expl(X, max_evals=self.max_evals)
            return _normalize_shap_values(exp.values)
        # kernel
        if self.background is None:
            raise ValueError("method='kernel' requires background")
        f = lambda z: model.predict_proba(z)[:, 1] if hasattr(model, "predict_proba") else model.predict(z)
        expl = shap.KernelExplainer(f, self.background)
        vals = expl.shap_values(X, nsamples=self.nsamples)
        return _normalize_shap_values(vals)


# Backward-compatible aliases
class Order1TreeSHAP(Explainer):
    def __init__(self, check_additivity: bool = False, feature_perturbation: str = "auto", model_output: str = "raw", **kwargs):
        self._engine = Order1Explainer(
            method="tree",
            check_additivity=check_additivity,
            feature_perturbation=feature_perturbation,
            model_output=model_output,
            **kwargs,
        )

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        return self._engine.compute(model, X, **kwargs)


class Order1PermutationSHAP(Explainer):
    def __init__(self, background: np.ndarray, max_evals: int = 2000, **kwargs):
        self._engine = Order1Explainer(method="permutation", background=background, max_evals=max_evals, **kwargs)

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        return self._engine.compute(model, X, **kwargs)


class Order1KernelSHAP(Explainer):
    def __init__(self, background: np.ndarray, nsamples: int = 200, **kwargs):
        self._engine = Order1Explainer(method="kernel", background=background, nsamples=nsamples, **kwargs)

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        return self._engine.compute(model, X, **kwargs)
