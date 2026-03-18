#!/usr/bin/env python3
"""
Jeu de coalitions (type Owen) par cluster ordre 1 – California Housing.

Version simple : on part des valeurs SHAP ordre 1 déjà calculées
(`figures/california_order1/shap_clusters.csv`) et on agrège ces contributions
par blocs de variables (coalitions) pour chaque cluster.

On ne recalcule pas Owen au sens strict via SHAP Partition (trop dépendant
de la version de la librairie), mais on obtient une vue très proche :
  - importance moyenne de chaque feature dans le cluster,
  - importance moyenne de chaque coalition = somme des SHAP moyens des
    features du bloc.

Résultats :
  figures/owen_winter_california_per_order1_cluster/cluster_k/
    - owen_summary_cluster_k.csv
    - owen_coalitions_bar.png
    - owen_features_bar.png

Une implémentation plus « fidèle » d'Owen/Winter pourra réutiliser la même
structure de sortie plus tard.
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


FEATURE_NAMES: List[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Coalitions fixées globalement (indices seront dérivés de FEATURE_NAMES)
COALITIONS: Dict[str, Sequence[str]] = {
    "spatial": ["Latitude", "Longitude"],
    "socio_eco": ["MedInc", "HouseAge"],
    "stock": ["AveRooms", "AveBedrms"],
    "density": ["Population", "AveOccup"],
}


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _aggregate_coalitions_from_shap(
    means_feat: np.ndarray,
) -> Dict[str, float]:
    """
    Agrège les valeurs d'Owen par coalition (moyenne sur les points, somme des features du bloc).
    Retourne un dict coalition -> valeur moyenne.
    """
    name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    coal_means: Dict[str, float] = {}
    for cname, vars_in_group in COALITIONS.items():
        idxs = [name_to_idx[v] for v in vars_in_group if v in name_to_idx]
        if not idxs:
            continue
        coal_means[cname] = float(means_feat[idxs].sum())
    return coal_means


def _plot_bar_coalitions(
    coal_means: Dict[str, float],
    path: str,
    title: str,
):
    if not coal_means:
        return
    names = list(coal_means.keys())
    vals = np.array([coal_means[n] for n in names])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color="#7aa2f7")
    ax.set_ylabel("Owen moyen (somme des features du bloc)")
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_bar_features(
    means_feat: np.ndarray,
    path: str,
    title: str,
):
    if means_feat.shape[0] != len(FEATURE_NAMES):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.argsort(means_feat)[::-1]
    names = [FEATURE_NAMES[i] for i in idx]
    vals = means_feat[idx]
    ax.bar(names, vals, color="#9ece6a")
    ax.set_ylabel("Owen moyen (par feature)")
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_owen_per_cluster(
    seed: int = 0,
    min_points_cluster: int = 20,
) -> None:
    """
    Calcule, pour chaque cluster ordre 1, l'importance moyenne :
      - de chaque feature (moyenne SHAP ordre 1),
      - de chaque coalition (somme des SHAP moyens des features du bloc).

    On part de figures/california_order1/shap_clusters.csv, déjà généré
    par scripts/california_housing_shap.py --order 1.
    """
    base = _base_dir()

    o1_csv = os.path.join(base, "figures/california_order1/shap_clusters.csv")
    if not os.path.isfile(o1_csv):
        raise FileNotFoundError(
            f"Fichier ordre 1 introuvable : {o1_csv}. "
            f"Lancez d'abord scripts/california_housing_shap.py --order 1."
        )

    df_o1 = pd.read_csv(o1_csv)
    if "cluster" not in df_o1.columns:
        raise ValueError("Le CSV ordre 1 doit contenir une colonne 'cluster'.")

    shap_cols = [f"shap_{name}" for name in FEATURE_NAMES]
    for c in shap_cols:
        if c not in df_o1.columns:
            raise ValueError(f"Colonne manquante dans le CSV ordre 1 : {c}")

    labels = df_o1["cluster"].values
    clusters = sorted(c for c in np.unique(labels) if c >= 0)

    out_parent = os.path.join(base, "figures/owen_winter_california_per_order1_cluster")
    os.makedirs(out_parent, exist_ok=True)

    summary_rows = []
    for c in clusters:
        mask = labels == c
        df_c = df_o1.loc[mask]
        if len(df_c) < min_points_cluster:
            print(f"cluster_{int(c)}: trop peu de points ({len(df_c)}), ignoré.")
            continue

        subdir = os.path.join(out_parent, f"cluster_{int(c)}")
        os.makedirs(subdir, exist_ok=True)

        print(f"cluster_{int(c)}: agrégation SHAP sur {len(df_c)} points ...")
        vals = df_c[shap_cols].values
        means_feat = vals.mean(axis=0)
        coal_means = _aggregate_coalitions_from_shap(means_feat)

        # CSV récapitulatif pour le cluster
        row = {"cluster": int(c), "n_points": int(len(df_c))}
        for j, name in enumerate(FEATURE_NAMES):
            row[f"owen_{name}"] = float(means_feat[j])
        for cname, val in coal_means.items():
            row[f"owen_coal_{cname}"] = float(val)
        summary_rows.append(row)

        df_cluster = pd.DataFrame([row])
        csv_path = os.path.join(subdir, f"owen_summary_cluster_{int(c)}.csv")
        df_cluster.to_csv(csv_path, index=False)

        # Figures
        _plot_bar_coalitions(
            coal_means,
            os.path.join(subdir, "owen_coalitions_bar.png"),
            title=f"Cluster {int(c)} – Owen par coalition",
        )
        _plot_bar_features(
            means_feat,
            os.path.join(subdir, "owen_features_bar.png"),
            title=f"Cluster {int(c)} – Owen par feature",
        )

    # CSV global (tous clusters)
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(
            os.path.join(out_parent, "owen_summary_all_clusters.csv"),
            index=False,
        )
        print("Résumé global Owen écrit dans", os.path.join(out_parent, "owen_summary_all_clusters.csv"))
    print("Terminé.")


def main():
    parser = argparse.ArgumentParser(
        description="Jeu de coalitions (agrégation SHAP) par cluster ordre 1 – California Housing."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-points-cluster",
        type=int,
        default=20,
        help="Taille minimale d'un cluster ordre 1 pour calculer Owen",
    )
    args = parser.parse_args()

    run_owen_per_cluster(
        seed=args.seed,
        min_points_cluster=args.min_points_cluster,
    )


if __name__ == "__main__":
    main()

