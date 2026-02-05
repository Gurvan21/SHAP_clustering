import numpy as np

def vectorize_order1(phi1: np.ndarray) -> np.ndarray:
    return np.asarray(phi1)

def vectorize_interactions(shap2: np.ndarray, include_diag: bool = False):
    """(n,p,p) -> (n,d) with upper triangle (optionally incl diag)."""
    shap2 = np.asarray(shap2)
    n, p, _ = shap2.shape
    iu = np.triu_indices(p, k=0 if include_diag else 1)
    Z = shap2[:, iu[0], iu[1]]
    return Z, iu
