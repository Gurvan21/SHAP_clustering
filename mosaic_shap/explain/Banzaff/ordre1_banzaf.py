from __future__ import annotations
from typing import Any, Callable, Optional
import numpy as np
from .base import Explainer


def _predict_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that maps (n,p) -> (n,) for regression or proba(class1) for binary classification."""
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]
    return lambda X: model.predict(X)


class Order1Banzhaf(Explainer):
    """
    Monte-Carlo estimation of first-order Banzhaf values.

    For each sample x and feature j:
        beta_j(x) = E_{S ⊆ F\{j} uniform} [ v(S ∪ {j}) - v(S) ]
    where v(S) is estimated by averaging model predictions over background completions
    for features not in S.
    """

    def __init__(
        self,
        background: np.ndarray,
        n_masks: int = 256,          # number of random subsets S per feature (per sample)
        n_bg: int = 32,              # number of background rows used to integrate missing features
        random_state: int = 0,
        normalize: bool = False,     # optional: enforce efficiency by rescaling contributions
    ):
        self.background = np.asarray(background)
        self.n_masks = int(n_masks)
        self.n_bg = int(n_bg)
        self.random_state = int(random_state)
        self.normalize = bool(normalize)

    def _v_of_S(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,              # (p,)
        mask: np.ndarray,           # (p,) bool, True => feature present (use x), False => absent (use bg)
        rng: np.random.Generator,
    ) -> float:
        """Estimate v(S) by Monte-Carlo over background completions."""
        B = self.background
        idx = rng.choice(B.shape[0], size=min(self.n_bg, B.shape[0]), replace=False)
        Bsub = B[idx]  # (n_bg, p)

        Xfill = Bsub.copy()
        Xfill[:, mask] = x[mask]     # keep present features from x
        preds = f(Xfill)             # (n_bg,)
        return float(np.mean(preds))

    def compute(self, model: Any, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X)
        n, p = X.shape
        f = _predict_fn(model)
        rng = np.random.default_rng(self.random_state)

        out = np.zeros((n, p), dtype=float)

        for i in range(n):
            x = X[i]
            # optional base value v(empty): all absent
            # base_mask = np.zeros(p, dtype=bool)

            for j in range(p):
                acc = 0.0

                # Sample subsets S ⊆ F\{j} uniformly:
                # Equivalent: for each k != j, include it with prob 1/2 independently.
                for _ in range(self.n_masks):
                    mask = rng.random(p) < 0.5
                    mask[j] = False  # ensure j not in S

                    vS = self._v_of_S(f, x, mask, rng)

                    mask_with_j = mask.copy()
                    mask_with_j[j] = True
                    vSj = self._v_of_S(f, x, mask_with_j, rng)

                    acc += (vSj - vS)

                out[i, j] = acc / self.n_masks

        if self.normalize:
            # Enforce efficiency approximately: scale so sum_j beta_j = f(x) - E_bg[f]
            # This is optional because Banzhaf doesn't guarantee efficiency by default.
            # Compute base as mean prediction over background rows (global baseline).
            base = float(np.mean(f(self.background)))
            fx = f(X).astype(float)  # (n,)
            denom = out.sum(axis=1)
            # avoid division by zero
            scale = np.ones_like(denom)
            nonzero = np.abs(denom) > 1e-12
            scale[nonzero] = (fx[nonzero] - base) / denom[nonzero]
            out = out * scale[:, None]

        return out
