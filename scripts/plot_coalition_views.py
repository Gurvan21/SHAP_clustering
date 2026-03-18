#!/usr/bin/env python3
"""
Visualisations globales pour les coalitions (type Owen) à partir de SHAP ordre 1.

On part de figures/california_order1/shap_clusters.csv et on construit :
  - une heatmap clusters × coalitions (moyenne des scores de coalition),
  - une vue UMAP couleur = coalition dominante (par point),
  - une carte Californie couleur = coalition dominante (par point).

Les coalitions sont les mêmes que dans run_owen_winter_per_cluster.py :
  spatial   = {Latitude, Longitude}
  socio_eco = {MedInc, HouseAge}
  stock     = {AveRooms, AveBedrms}
  density   = {Population, AveOccup}

Sorties dans figures/owen_winter_california_per_order1_cluster/ :
  - heatmap_cluster_vs_coalitions.png
  - umap_coalitions.png
  - map_coalitions.png
"""

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.data.Data_Pedagogique import make_dataset_Housing_California  # noqa: F401
from california_housing_shap import FEATURE_NAMES  # réutilise la même liste
from california_housing_shap import SUBDIR_ORDER1, CSV_NAME  # pour localiser le CSV


COALITIONS: Dict[str, List[str]] = {
    "spatial": ["Latitude", "Longitude"],
    "socio_eco": ["MedInc", "HouseAge"],
    "stock": ["AveRooms", "AveBedrms"],
    "density": ["Population", "AveOccup"],
}


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _compute_coalition_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    À partir des colonnes shap_* par feature, ajoute pour chaque coalition
    une colonne coal_score_<nom> = somme des shap_... de ses variables.
    """
    df = df.copy()
    for cname, vars_in_group in COALITIONS.items():
        cols = [f"shap_{v}" for v in vars_in_group if f"shap_{v}" in df.columns]
        if not cols:
            continue
        df[f"coal_score_{cname}"] = df[cols].sum(axis=1)
    return df


def _add_dominant_coalition(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne coalition_dom = coalition avec score absolu maximal pour chaque point."""
    df = df.copy()
    coal_cols = [c for c in df.columns if c.startswith("coal_score_")]
    if not coal_cols:
        return df
    scores = df[coal_cols].values
    # On regarde l'importance en valeur absolue
    idx = np.argmax(np.abs(scores), axis=1)
    names = [c.replace("coal_score_", "") for c in coal_cols]
    df["coalition_dom"] = [names[i] for i in idx]
    return df


def plot_heatmap_cluster_vs_coalitions(df: pd.DataFrame, path: str):
    """Heatmap : lignes = clusters ordre 1, colonnes = coalitions, valeur = moyenne des scores."""
    labels = np.asarray(df["cluster"])
    clusters = sorted(c for c in np.unique(labels) if c >= 0)
    coal_cols = [c for c in df.columns if c.startswith("coal_score_")]
    if not clusters or not coal_cols:
        return
    col_labels = [c.replace("coal_score_", "") for c in coal_cols]

    M = np.zeros((len(clusters), len(coal_cols)))
    for i, c in enumerate(clusters):
        mask = labels == c
        M[i] = df.loc[mask, coal_cols].mean(axis=0).values

    fig, ax = plt.subplots(figsize=(6, max(4, 0.5 * len(clusters))))
    vmax = np.abs(M).max()
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Score moyen de coalition (somme SHAP)")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center")
    ax.set_title("Clusters ordre 1 × coalitions (moyenne des scores)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_umap_coalitions(df: pd.DataFrame, path: str):
    """UMAP (ordre 1) couleur = coalition dominante."""
    if "coalition_dom" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = df["coalition_dom"].values
    uniq = pd.unique(labels)
    for c in uniq:
        mask = labels == c
        ax.scatter(
            df.loc[mask, "umap_1"],
            df.loc[mask, "umap_2"],
            s=8,
            alpha=0.7,
            label=str(c),
        )
    ax.set_title("UMAP ordre 1 – couleur = coalition dominante (|score|)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_map_coalitions(df: pd.DataFrame, path: str):
    """
    Carte Californie couleur = coalition dominante (scatter simple).
    """
    if "coalition_dom" not in df.columns:
        return
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = df["coalition_dom"].values
    uniq = pd.unique(labels)
    for c in uniq:
        mask = labels == c
        ax.scatter(
            df.loc[mask, "Longitude"],
            df.loc[mask, "Latitude"],
            s=8,
            alpha=0.7,
            label=str(c),
        )
    ax.set_title("Carte Californie – couleur = coalition dominante (|score|)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
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

    # Vérifier la présence des colonnes SHAP
    missing = [f"shap_{f}" for f in FEATURE_NAMES if f"shap_{f}" not in df.columns]
    if missing:
        raise ValueError(f"Colonnes SHAP manquantes : {missing}")

    df = _compute_coalition_scores(df)
    df = _add_dominant_coalition(df)

    out_dir = os.path.join(base, "figures/owen_winter_california_per_order1_cluster")
    os.makedirs(out_dir, exist_ok=True)

    plot_heatmap_cluster_vs_coalitions(
        df, os.path.join(out_dir, "heatmap_cluster_vs_coalitions.png")
    )
    plot_umap_coalitions(
        df, os.path.join(out_dir, "umap_coalitions.png")
    )
    plot_map_coalitions(
        df, os.path.join(out_dir, "map_coalitions.png")
    )
    print("Figures de coalitions créées dans", out_dir)


if __name__ == "__main__":
    main()

