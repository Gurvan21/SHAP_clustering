import numpy as np

from typing import List, Dict
from mosaic_shap.explain.Winter_Shap.winter_explainer import WINTERExplainer



def winter_by_region(
    X: np.ndarray,
    region_labels: np.ndarray,
    model,
    coarse_groups: List[List[int]],
    fine_groups: List[List[int]],
    background: np.ndarray,
    n_permutations: int = 64,
    min_region_size: int = 10,
    feature_names = None
) -> Dict:
    """
    Calcule les valeurs de Winter pour chaque région / parcelle.

    Parameters
    ----------
    X : np.ndarray, shape (N, M)
    region_labels : np.ndarray, shape (N,)
        Labels de région (-1 = bruit/non assigné).

    Returns
    -------
    results : dict
        {
          'regional_means': np.ndarray (R, M) — valeurs moyennes par région,
          'heterogeneity':  np.ndarray (M,)   — std des valeurs moyennes,
          'region_sizes':   dict,
        }

    Notes
    -----
    À intégrer dans mosaic_shap/regional.py
    """
    unique_regions = [r for r in np.unique(region_labels) if r >= 0]
    regional_means = []
    region_sizes   = {}

    winter_exp = WINTERExplainer(
        model=model,
        coarse_groups=coarse_groups,
        fine_groups=fine_groups,
        background=background,
        n_permutations=n_permutations,
        feature_names = feature_names
    )

    for r in unique_regions:
        idx_r = np.where(region_labels == r)[0]
        if len(idx_r) < min_region_size:
            print(f"  Région {r} : trop petite ({len(idx_r)} points), ignorée")
            continue

        wv_r = winter_exp.shap_values(X[idx_r])
        regional_means.append(np.abs(wv_r).mean(0))
        region_sizes[r] = len(idx_r)

    regional_means = np.array(regional_means)  # (R, M)

    # ── Fix : gérer le cas 0 ou 1 région ─────────────────────────────────────────
    if len(regional_means) == 0:
        heterogeneity = np.zeros(self.M if hasattr(self, 'M') else X.shape[1])
    elif len(regional_means) == 1:
        heterogeneity = np.zeros(regional_means.shape[1])
    else:
        heterogeneity = regional_means.std(0)   # (M,) — cas normal

    return {
        'regional_means': regional_means,
        'heterogeneity': heterogeneity,
        'region_sizes': region_sizes,
        'n_regions': len(regional_means),
    }
