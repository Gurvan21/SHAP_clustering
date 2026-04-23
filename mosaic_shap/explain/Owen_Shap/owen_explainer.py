import numpy as np
import pandas as pd


class OWENExplainer:
    def __init__(self, model, groups, background, n_permutations=64, feature_names = None):
        self.model = model
        self.groups = [list(g) for g in groups]
        self.background = np.array(background, dtype=float)
        self.n_permutations = n_permutations
        self.M = background.shape[1]
        self.K = len(groups)
        self.feature_names = feature_names

        all_features = sorted([f for g in groups for f in g])
        assert all_features == list(range(self.M))
        self._feat_to_group = {f: k for k, g in enumerate(groups) for f in g}

    def _v_batch(self, x: np.ndarray, list_of_active: list) -> np.ndarray:
        """
        Évalue v(S) pour TOUTES les coalitions d'un coup en un seul predict().
        list_of_active : liste de listes d'indices actifs
        Retourne un vecteur de valeurs, une par coalition.
        """
        n_bg   = len(self.background)
        n_coal = len(list_of_active)
        X_big  = np.tile(self.background, (n_coal, 1))
        for k, active in enumerate(list_of_active):
            start = k * n_bg
            end   = start + n_bg
            for f in active:
                X_big[start:end, f] = x[f]
        
        # ── Wrap en DataFrame pour correspondre au format d'entraînement ──
        X_big_df = pd.DataFrame(X_big, columns=self.feature_names)
        preds = self.model.predict(X_big_df)
        return preds.reshape(n_coal, n_bg).mean(axis=1)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        N = X.shape[0]
        owen_values = np.zeros((N, self.M))

        for n in range(N):
            x = X[n]
            contributions = np.zeros(self.M)

            for _ in range(self.n_permutations):
                group_order = np.random.permutation(self.K)
                feature_order_by_group = [
                    np.random.permutation(self.groups[k]).tolist()
                    for k in range(self.K)
                ]

                # ── Collecter TOUS les couples (before, after) de la permutation
                pairs_before = []
                pairs_after  = []
                feat_sequence = []

                for pos_k, k in enumerate(group_order):
                    prev_groups_features = [
                        f for kk in group_order[:pos_k] for f in self.groups[kk]
                    ]
                    intra_active = []
                    for feat in feature_order_by_group[k]:
                        before = prev_groups_features + intra_active
                        after  = before + [feat]
                        pairs_before.append(list(before))
                        pairs_after.append(list(after))
                        feat_sequence.append(feat)
                        intra_active.append(feat)

                # ── Un seul appel predict() pour toutes les coalitions
                all_coalitions = pairs_before + pairs_after
                all_vals = self._v_batch(x, all_coalitions)

                n_pairs = len(pairs_before)
                vals_before = all_vals[:n_pairs]
                vals_after  = all_vals[n_pairs:]

                for feat, vb, va in zip(feat_sequence, vals_before, vals_after):
                    contributions[feat] += va - vb

            owen_values[n] = contributions / self.n_permutations
            if (n + 1) % 5 == 0:
                print(f"  {n+1}/{N} observations traitées...")

        return owen_values
print("OWENExplainer défini")