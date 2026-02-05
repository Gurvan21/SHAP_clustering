from __future__ import annotations
from typing import Any
import numpy as np
import shap
from .base import Explainer

class Order1TreeSHAP(Explainer):
    def __init__(self, check_additivity: bool = False, feature_perturbation: str | None = None, model_output: str = "raw"):
        self.check_additivity = check_additivity
        self.feature_perturbation = feature_perturbation
        self.model_output = model_output

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        expl = shap.TreeExplainer(model, feature_perturbation=self.feature_perturbation, model_output=self.model_output)
        vals = expl.shap_values(X, check_additivity=self.check_additivity)
        if isinstance(vals, list):
            vals = vals[1]
        return np.asarray(vals)

class Order1PermutationSHAP(Explainer):
    def __init__(self, background: np.ndarray, max_evals: int = 2000):
        self.background = background
        self.max_evals = max_evals

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        expl = shap.Explainer(model, self.background, algorithm="permutation")
        exp = expl(X, max_evals=self.max_evals)
        vals = exp.values
        if isinstance(vals, list):
            vals = vals[1]
        return np.asarray(vals)

class Order1KernelSHAP(Explainer):
    def __init__(self, background: np.ndarray, nsamples: int = 200):
        self.background = background
        self.nsamples = nsamples

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        f = lambda z: model.predict_proba(z)[:,1] if hasattr(model, "predict_proba") else model.predict(z)
        expl = shap.KernelExplainer(f, self.background)
        vals = expl.shap_values(X, nsamples=self.nsamples)
        if isinstance(vals, list):
            vals = vals[1]
        return np.asarray(vals)
