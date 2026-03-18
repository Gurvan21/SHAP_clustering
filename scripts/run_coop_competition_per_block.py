#!/usr/bin/env python3
"""
Indices de coopération / compétition à l'intérieur des blocs Owen, par cluster ordre 1.

Idée :
  - On considère une décision binaire dérivée du modèle de régression :
        d(x) = 1 si prédiction >= médiane globale des prédictions, sinon 0.
  - Pour chaque coalition (bloc Owen) C et pour chaque paire (i,j) dans C,
    et pour chaque point x d'un cluster :
        v(∅)   : décision quand ni i ni j ne sont actifs (remplacés par baseline),
        v({i}) : décision quand seule i est active,
        v({j}) : décision quand seule j est active,
        v({i,j}): décision quand i et j sont actives.
    On compte :
        - Coopération : v({i,j}) != v(∅) ET v({i}) == v(∅) ET v({j}) == v(∅)
                        (les deux ensemble sont nécessaires pour changer)
        - Compétition : v({i,j}) != v(∅) ET (v({i}) != v(∅) OU v({j}) != v(∅))
                        (au moins une seule suffit à changer)

  - On agrège ces comptes sur les points et sur les paires (i,j) d'un bloc C
    pour obtenir, par coalition et par cluster :
        Coop(C), Comp(C), Coop_rate, Comp_rate.

Sorties :
  figures/coop_comp_california_per_order1_cluster/cluster_k/
    - coop_comp_cluster_k.csv
    - coop_comp_cluster_k.png  (barplot Coop vs Comp par coalition)
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from california_housing_shap import (
    FEATURE_NAMES,
    SUBDIR_ORDER1,
    CSV_NAME,
    FEAT_PREFIX,
    _fit_model,  # type: ignore
)


COALITIONS: Dict[str, List[str]] = {
    "spatial": ["Latitude", "Longitude"],
    "socio_eco": ["MedInc", "HouseAge"],
    "stock": ["AveRooms", "AveBedrms"],
    "density": ["Population", "AveOccup"],
}


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _build_index_pairs() -> Dict[str, List[Tuple[int, int]]]:
    """Pour chaque coalition, retourne la liste des paires (i,j) d'indices de features."""
    name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    coal_pairs: Dict[str, List[Tuple[int, int]]] = {}
    for cname, vars_in_group in COALITIONS.items():
        idxs = [name_to_idx[v] for v in vars_in_group if v in name_to_idx]
        pairs: List[Tuple[int, int]] = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((idxs[a], idxs[b]))
        if pairs:
            coal_pairs[cname] = pairs
    return coal_pairs


def _decision_threshold(model, X: np.ndarray) -> float:
    """Seuil = médiane des prédictions du modèle sur X."""
    y_hat = model.predict(X)
    return float(np.median(y_hat))


def _decision(model, x: np.ndarray, thresh: float) -> int:
    """Décision binaire à partir de la prédiction du modèle et du seuil."""
    y = float(model.predict(x.reshape(1, -1))[0])
    return int(y >= thresh)


def _coop_comp_for_pair(
    model,
    X_cluster: np.ndarray,
    idx_i: int,
    idx_j: int,
    base: np.ndarray,
    thresh: float,
    max_points: int,
) -> Tuple[int, int, int]:
    """
    Calcule (coop_count, comp_count, total_events) pour une paire (i,j) dans un cluster.
    total_events = nombre de points pour lesquels v({i,j}) != v(∅).
    """
    n = X_cluster.shape[0]
    rng = np.random.default_rng(0)
    if n > max_points:
        sel = rng.choice(n, size=max_points, replace=False)
        X_use = X_cluster[sel]
    else:
        X_use = X_cluster

    coop = 0
    comp = 0
    total = 0

    for xrow in X_use:
        # v(∅)
        x_empty = base.copy()
        v_empty = _decision(model, x_empty, thresh)

        # v({i})
        x_i = base.copy()
        x_i[idx_i] = xrow[idx_i]
        v_i = _decision(model, x_i, thresh)

        # v({j})
        x_j = base.copy()
        x_j[idx_j] = xrow[idx_j]
        v_j = _decision(model, x_j, thresh)

        # v({i,j})
        x_ij = base.copy()
        x_ij[idx_i] = xrow[idx_i]
        x_ij[idx_j] = xrow[idx_j]
        v_ij = _decision(model, x_ij, thresh)

        if v_ij == v_empty:
            continue
        total += 1

        # Coopération : seul le couple change la décision
        if v_i == v_empty and v_j == v_empty and v_ij != v_empty:
            coop += 1
        # Compétition : au moins un seul suffit à changer
        elif (v_i != v_empty) or (v_j != v_empty):
            comp += 1

    return coop, comp, total


def run_coop_comp_per_cluster(
    seed: int = 0,
    max_points_per_pair: int = 200,
    min_points_cluster: int = 50,
) -> None:
    base = _base_dir()
    csv_path = os.path.join(base, SUBDIR_ORDER1, CSV_NAME)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV ordre 1 introuvable : {csv_path}. "
            "Lancez d'abord scripts/california_housing_shap.py --order 1."
        )

    df = pd.read_csv(csv_path)
    if "cluster" not in df.columns:
        raise ValueError("Le CSV ordre 1 doit contenir une colonne 'cluster'.")

    feat_cols = [c for c in df.columns if c.startswith(FEAT_PREFIX)]
    if len(feat_cols) != len(FEATURE_NAMES):
        raise ValueError(
            f"Nombre de colonnes feat_* inattendu ({len(feat_cols)}). "
            f"Attendu {len(FEATURE_NAMES)} pour {FEATURE_NAMES}."
        )

    X = df[feat_cols].values
    y = df["price"].values
    labels = df["cluster"].values
    clusters = sorted(c for c in np.unique(labels) if c >= 0)

    # Modèle identique à celui utilisé pour SHAP
    model = _fit_model(X, y, seed)

    # Baseline globale = moyenne des features sur tout le jeu
    base_vec = X.mean(axis=0)
    thresh = _decision_threshold(model, X)

    coal_pairs = _build_index_pairs()

    out_parent = os.path.join(base, "figures/coop_comp_california_per_order1_cluster")
    os.makedirs(out_parent, exist_ok=True)

    for c in clusters:
        mask = labels == c
        X_c = X[mask]
        if len(X_c) < min_points_cluster:
            print(f"cluster_{int(c)}: trop peu de points ({len(X_c)}), ignoré.")
            continue

        subdir = os.path.join(out_parent, f"cluster_{int(c)}")
        os.makedirs(subdir, exist_ok=True)
        print(f"cluster_{int(c)}: calcul Coop/Comp sur {len(X_c)} points ...")

        rows = []
        for cname, pairs in coal_pairs.items():
            total_coop = 0
            total_comp = 0
            total_events = 0
            for (i, j) in pairs:
                coop, comp, tot = _coop_comp_for_pair(
                    model,
                    X_c,
                    idx_i=i,
                    idx_j=j,
                    base=base_vec,
                    thresh=thresh,
                    max_points=max_points_per_pair,
                )
                total_coop += coop
                total_comp += comp
                total_events += tot

            if total_events == 0:
                coop_rate = 0.0
                comp_rate = 0.0
            else:
                coop_rate = total_coop / total_events
                comp_rate = total_comp / total_events

            rows.append(
                {
                    "cluster": int(c),
                    "coalition": cname,
                    "coop": total_coop,
                    "comp": total_comp,
                    "events": total_events,
                    "coop_rate": coop_rate,
                    "comp_rate": comp_rate,
                }
            )

        df_cluster = pd.DataFrame(rows)
        csv_out = os.path.join(subdir, f"coop_comp_cluster_{int(c)}.csv")
        df_cluster.to_csv(csv_out, index=False)

        # Barplot Coop/Comp par coalition
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(df_cluster))
        width = 0.35
        ax.bar(x - width / 2, df_cluster["coop_rate"], width, label="Coopération", color="#9ece6a")
        ax.bar(x + width / 2, df_cluster["comp_rate"], width, label="Compétition", color="#f7768e")
        ax.set_xticks(x)
        ax.set_xticklabels(df_cluster["coalition"], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion d'événements (0–1)")
        ax.set_title(f"Cluster {int(c)} – Coopération vs Compétition par coalition")
        ax.legend()
        plt.tight_layout()
        fig_path = os.path.join(subdir, f"coop_comp_cluster_{int(c)}.png")
        plt.savefig(fig_path, dpi=180)
        plt.close()

    print("Terminé. Résultats dans", out_parent)


def main():
    parser = argparse.ArgumentParser(
        description="Indices de coopération / compétition par bloc Owen et par cluster ordre 1."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-points-per-pair",
        type=int,
        default=200,
        help="Nombre max de points utilisés par paire (i,j) dans un cluster.",
    )
    parser.add_argument(
        "--min-points-cluster",
        type=int,
        default=50,
        help="Nombre minimal de points dans un cluster pour calculer les indices.",
    )
    args = parser.parse_args()

    run_coop_comp_per_cluster(
        seed=args.seed,
        max_points_per_pair=args.max_points_per_pair,
        min_points_cluster=args.min_points_cluster,
    )


if __name__ == "__main__":
    main()

