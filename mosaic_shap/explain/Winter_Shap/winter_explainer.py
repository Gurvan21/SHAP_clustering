import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import pdist
from itertools import combinations, permutations
from math import factorial
from typing import List, Dict, Tuple, Optional

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import shap
import scipy

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform



class WINTERExplainer:
    """
    Calcule les valeurs de Winter pour un modèle arbitraire.

    Implémentation par Monte-Carlo à 3 niveaux.

    Parameters
    ----------
    model : callable
        Modèle avec une méthode .predict(X).
    coarse_groups : list of list of int
        Partition grossière. Ex : [[0,1,2,3], [4,5,6,7]]
    fine_groups : list of list of int
        Partition fine COMPLÈTE. Ex : [[0,1], [2,3], [4,5], [6,7]]
        Chaque groupe fin doit être inclus dans exactement un groupe grossier.
    background : np.ndarray, shape (n_bg, M)
    n_permutations : int

    Notes
    -----
    Relation avec Owen :
        Owen = Winter avec un seul niveau de partition (coarse = fine)
    """

    def __init__(self, model, coarse_groups: List[List[int]],
                 fine_groups: List[List[int]],
                 background: np.ndarray, n_permutations: int = 256, 
                 feature_names = None):
        self.model = model
        self.coarse_groups = [list(g) for g in coarse_groups]
        self.fine_groups   = [list(g) for g in fine_groups]
        self.background    = np.array(background)
        self.n_permutations = n_permutations
        self.M  = background.shape[1]
        self.KC = len(coarse_groups)  # nombre de groupes grossiers
        self.KF = len(fine_groups)    # nombre de groupes fins
        self.feature_names = feature_names

        # Vérifications de cohérence
        all_fine   = sorted([f for g in fine_groups   for f in g])
        all_coarse = sorted([f for g in coarse_groups for f in g])
        assert all_fine   == list(range(self.M)), "fine_groups doit être une partition complète"
        assert all_coarse == list(range(self.M)), "coarse_groups doit être une partition complète"

        # Mapping feature → groupe fin → groupe grossier
        self._feat_to_fine   = {}
        self._feat_to_coarse = {}
        self._fine_to_coarse = {}

        for kf, fg in enumerate(fine_groups):
            for f in fg:
                self._feat_to_fine[f] = kf

        for kc, cg in enumerate(coarse_groups):
            for f in cg:
                self._feat_to_coarse[f] = kc

        # Pour chaque groupe grossier : quels groupes fins contient-il ?
        self._coarse_to_fines: Dict[int, List[int]] = {kc: [] for kc in range(self.KC)}
        for kf, fg in enumerate(fine_groups):
            # On détermine le groupe grossier d'appartenance via le premier élément
            kc = self._feat_to_coarse[fg[0]]
            assert all(self._feat_to_coarse[f] == kc for f in fg), (
                f"Le groupe fin {kf} chevauche plusieurs groupes grossiers !"
            )
            self._coarse_to_fines[kc].append(kf)
            self._fine_to_coarse[kf] = kc

    def _v_batch(self, x: np.ndarray, list_of_active: list) -> np.ndarray:
        n_bg   = len(self.background)
        n_coal = len(list_of_active)
        X_big  = np.tile(self.background, (n_coal, 1)).astype(float)
        for k, active in enumerate(list_of_active):
            start, end = k * n_bg, (k + 1) * n_bg
            for f in active:
                X_big[start:end, f] = x[f]

        # ── Wrap en DataFrame si feature_names disponible ──
        if self.feature_names is not None:
            X_big = pd.DataFrame(X_big, columns=self.feature_names)

        preds = self.model.predict(X_big)
        return preds.reshape(n_coal, n_bg).mean(axis=1)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les valeurs de Winter pour chaque observation.

        Returns
        -------
        winter_values : np.ndarray, shape (N, M)
        """
        X = np.array(X)
        N = X.shape[0]
        winter_values = np.zeros((N, self.M))

        for n in range(N):
            x = X[n]
            contributions = np.zeros(self.M)

            for _ in range(self.n_permutations):
                # ── Niveau 1 : ordre des groupes grossiers ───────────────────
                coarse_order = np.random.permutation(self.KC)

                # ── Niveau 2 : ordre des groupes fins au sein de chaque
                #              groupe grossier ────────────────────────────────
                fine_order_by_coarse = {
                    kc: np.random.permutation(self._coarse_to_fines[kc]).tolist()
                    for kc in range(self.KC)
                }

                # ── Niveau 3 : ordre des features au sein de chaque
                #              groupe fin ──────────────────────────────────────
                feat_order_by_fine = {
                    kf: np.random.permutation(self.fine_groups[kf]).tolist()
                    for kf in range(self.KF)
                }

                # ── Parcours selon la structure hiérarchique ──────────────────
                active = []

                for pos_c, kc in enumerate(coarse_order):
                    # Features des groupes grossiers PRÉCÉDENTS
                    prev_coarse_feats = [
                        f for kc2 in coarse_order[:pos_c]
                        for f in self.coarse_groups[kc2]
                    ]

                    # Parcours des groupes fins du groupe grossier courant
                    for pos_f, kf in enumerate(fine_order_by_coarse[kc]):
                        # Features des groupes fins précédents (dans ce groupe grossier)
                        prev_fine_feats = [
                            f for kf2 in fine_order_by_coarse[kc][:pos_f]
                            for f in self.fine_groups[kf2]
                        ]

                        # Parcours des features dans le groupe fin courant
                        intra_fine_active = []
                        pairs_before, pairs_after, feat_seq = [], [], []
                        for feat in feat_order_by_fine[kf]:
                            before = prev_coarse_feats + prev_fine_feats + intra_fine_active
                            after  = before + [feat]
                            pairs_before.append(list(before))
                            pairs_after.append(list(after))
                            feat_seq.append(feat)
                            intra_fine_active.append(feat)

                        all_vals = self._v_batch(x, pairs_before + pairs_after)
                        n_pairs  = len(pairs_before)
                        for feat, vb, va in zip(feat_seq, all_vals[:n_pairs], all_vals[n_pairs:]):
                            contributions[feat] += va - vb

            winter_values[n] = contributions / self.n_permutations

        return winter_values
