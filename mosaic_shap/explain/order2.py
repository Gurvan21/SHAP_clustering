from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import shap
from .base import Explainer, predict_score

@dataclass
class InteractionResult:
    values: np.ndarray  # (n,p,p)

class Order2TreeSHAPInteractions(Explainer):
    def compute(self, model: Any, X: np.ndarray, **kwargs) -> InteractionResult:
        expl = shap.TreeExplainer(model)
        shap2 = expl.shap_interaction_values(X)
        if isinstance(shap2, list):
            shap2 = shap2[1]
        if shap2.ndim == 4:
            shap2 = shap2[:, :, :, 1]
        return InteractionResult(values=np.asarray(shap2))

class Order2MonteCarloInteractions(Explainer):
    """Model-agnostic SHAP-IQ-style subset estimator for pairwise interactions."""
    def __init__(self, n_perms: int = 80, baseline: str = "mean", random_state: int = 0):
        self.n_perms = n_perms
        self.baseline = baseline
        self.random_state = random_state

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> InteractionResult:
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)
        n, p = X.shape
        base = X.mean(axis=0) if self.baseline == "mean" else np.zeros(p)

        def v(x_row: np.ndarray, mask: np.ndarray) -> float:
            x = base.copy()
            x[mask] = x_row[mask]
            return float(predict_score(model, x.reshape(1,-1))[0])

        out = np.zeros((n, p, p), dtype=float)
        for r in range(n):
            xrow = X[r]
            for _ in range(self.n_perms):
                S = rng.random(p) < 0.5
                for i in range(p):
                    for j in range(i+1, p):
                        if S[i] or S[j]:
                            continue
                        vS = v(xrow, S)
                        m_i = S.copy(); m_i[i] = True
                        m_j = S.copy(); m_j[j] = True
                        m_ij = S.copy(); m_ij[i] = True; m_ij[j] = True
                        vSi = v(xrow, m_i)
                        vSj = v(xrow, m_j)
                        vSij = v(xrow, m_ij)
                        inter = vSij - vSi - vSj + vS
                        out[r, i, j] += inter
                        out[r, j, i] += inter
            out[r] /= self.n_perms
        return InteractionResult(values=out)

class Order2RegressionInteractions(Explainer):
    """Model-agnostic regression surrogate over coalition masks (SHAP-IQ-style)."""
    def __init__(self, n_masks: int = 400, baseline: str = "mean", random_state: int = 0, ridge: float = 1e-6):
        self.n_masks = n_masks
        self.baseline = baseline
        self.random_state = random_state
        self.ridge = ridge

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> InteractionResult:
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)
        n, p = X.shape
        base = X.mean(axis=0) if self.baseline == "mean" else np.zeros(p)

        def v(x_row: np.ndarray, mask: np.ndarray) -> float:
            x = base.copy()
            x[mask] = x_row[mask]
            return float(predict_score(model, x.reshape(1,-1))[0])

        out = np.zeros((n, p, p), dtype=float)
        for r in range(n):
            xrow = X[r]
            masks = rng.random((self.n_masks, p)) < 0.5
            y = np.array([v(xrow, masks[k]) for k in range(self.n_masks)])
            d = 1 + p + p*(p-1)//2
            A = np.zeros((self.n_masks, d), dtype=float)
            A[:,0] = 1.0
            A[:,1:1+p] = masks.astype(float)
            col = 1+p
            for i in range(p):
                for j in range(i+1, p):
                    A[:,col] = (masks[:,i] & masks[:,j]).astype(float)
                    col += 1
            ATA = A.T @ A
            ATA.flat[::ATA.shape[0]+1] += self.ridge
            w = np.linalg.solve(ATA, A.T @ y)
            col = 1+p
            for i in range(p):
                for j in range(i+1, p):
                    out[r,i,j] = w[col]
                    out[r,j,i] = w[col]
                    col += 1
        return InteractionResult(values=out)
