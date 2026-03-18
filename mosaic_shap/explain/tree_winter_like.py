from __future__ import annotations
from typing import List, Tuple
import numpy as np


class WinterLikeTreeAttributor:
    """
    Approximation Winter-like pour modèles d'arbres additifs (GBDT sklearn).

    Pour chaque arbre et chaque x:
      - on suit le chemin actif jusqu'à la feuille,
      - on récupère les valeurs de noeud (moyennes locales),
      - on décompose la différence feuille - racine le long du chemin,
      - on attribue chaque incrément à la feature du split.

    Résultat:
      - contributions par feature, baseline_globale, prédiction.
    """

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.trees_ = self._extract_trees()
        # baseline arbre = valeur à la racine
        self.tree_baselines_ = np.array(
            [t.tree_.value[0, 0] for t in self.trees_], dtype=float
        )
        self.global_baseline_ = float(self.tree_baselines_.sum())

    def _extract_trees(self) -> List:
        trees = []
        if hasattr(self.model, "estimators_"):
            est = self.model.estimators_
            if est.ndim == 2:
                for row in est:
                    trees.append(row[0])
            else:
                for t in est:
                    trees.append(t)
        else:
            raise TypeError("Modèle non supporté: pas d'attribut estimators_.")
        return trees

    def _tree_path_contribs(self, tree, x: np.ndarray) -> Tuple[np.ndarray, float]:
        t = tree.tree_
        n_features = len(self.feature_names)
        contribs = np.zeros(n_features, dtype=float)

        # chemin actif
        node_indicator = tree.decision_path(x.reshape(1, -1))
        node_index = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        # valeurs de noeud (scalaires)
        values = t.value[node_index, 0, 0]  # shape (L,)

        for k in range(len(node_index) - 1):
            node_id = node_index[k]
            feat_idx = t.feature[node_id]
            if feat_idx < 0:
                continue
            delta = values[k + 1] - values[k]
            contribs[feat_idx] += delta

        leaf_value = float(values[-1])
        return contribs, leaf_value

    def explain_instance(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        x = np.asarray(x, dtype=float)
        n_features = len(self.feature_names)
        total = np.zeros(n_features, dtype=float)

        for tree in self.trees_:
            c_t, _ = self._tree_path_contribs(tree, x)
            total += c_t

        pred = float(self.model.predict(x.reshape(1, -1))[0])
        return total, self.global_baseline_, pred

    def explain_batch(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        assert n_features == len(self.feature_names)

        all_contribs = np.zeros((n_samples, n_features), dtype=float)
        preds = self.model.predict(X)

        for i in range(n_samples):
            x = X[i]
            total = np.zeros(n_features, dtype=float)
            for tree in self.trees_:
                c_t, _ = self._tree_path_contribs(tree, x)
                total += c_t
            all_contribs[i] = total

        return all_contribs, self.global_baseline_, preds

    def max_efficiency_error(self, X: np.ndarray) -> float:
        contribs, baseline, preds = self.explain_batch(X)
        approx = baseline + contribs.sum(axis=1)
        diff = np.abs(approx - preds)
        return float(diff.max())

