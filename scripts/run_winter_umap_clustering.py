#!/usr/bin/env python3
"""
UMAP + clustering sur les contributions Winter-like (approximatives) du modèle California Housing.

Entrée :
  - figures/california_order1/shap_clusters.csv
    avec colonnes winter_MedInc, ..., winter_Longitude.

Sorties (dans figures/california_winter/):
  - winter_umap_cluster.png       (UMAP couleur = cluster)
  - winter_umap_price.png         (UMAP couleur = prix)
  - winter_map_cluster.png        (carte CA couleur = cluster)
  - winter_map_price.png          (carte CA couleur = prix)
  - winter_heatmap_cluster_vs_feature.png (heatmap cluster × winter_feature)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.reduction import create_reducer
from mosaic_shap.clustering import create_clusterer
from california_housing_shap import _base_dir, FEATURE_NAMES
from california_housing_shap import (
    plot_umap_price,
    plot_umap_cluster,
    plot_map_cluster,
    plot_map_price,
)


def main():
    base = _base_dir()
    csv_path = os.path.join(base, "figures/california_order1/shap_clusters.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV ordre 1 introuvable : {csv_path}. "
            "Lancez d'abord scripts/california_housing_shap.py --order 1."
        )

    df = pd.read_csv(csv_path)
    winter_cols = [f"winter_{f}" for f in FEATURE_NAMES]
    for c in winter_cols:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}. Recalculez avec le script ordre 1.")

    # Centrage / normalisation simple des winter_* pour stabiliser UMAP / clustering
    Z = df[winter_cols].values
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)

    reducer = create_reducer("umap", n_components=2, random_state=0)
    Z_red = reducer.fit_transform(Z)

    # HDBSCAN un peu plus fin que l'ordre 1 SHAP : on accepte des clusters plus petits
    clusterer = create_clusterer("hdbscan", min_cluster_size=50, min_samples=10)
    labels = clusterer.fit_predict(Z_red)

    df["winter_umap_1"] = Z_red[:, 0]
    df["winter_umap_2"] = Z_red[:, 1]
    df["winter_cluster"] = labels

    out_dir = os.path.join(base, "figures/california_winter")
    os.makedirs(out_dir, exist_ok=True)

    # UMAP prix / cluster
    def _plot_umap_price_winter(df_local: pd.DataFrame, path: str):
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            df_local["winter_umap_1"],
            df_local["winter_umap_2"],
            c=df_local["price"],
            s=8,
            cmap="viridis",
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
        ax.set_title("UMAP Winter-like – couleur = prix")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def _plot_umap_cluster_winter(df_local: pd.DataFrame, path: str):
        fig, ax = plt.subplots(figsize=(7, 6))
        labels_loc = df_local["winter_cluster"].values
        u = np.unique(labels_loc)
        u = np.concatenate([u[u >= 0], u[u < 0]])
        for c in u:
            mask = labels_loc == c
            ax.scatter(
                df_local.loc[mask, "winter_umap_1"],
                df_local.loc[mask, "winter_umap_2"],
                s=8,
                alpha=0.7,
                label=f"Cluster {int(c)}",
            )
        ax.set_title("UMAP Winter-like – couleur = cluster")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    # Carte cluster / prix (en réutilisant les fonctions existantes)
    def _plot_map_cluster_winter(df_local: pd.DataFrame, path: str):
        tmp = df_local.copy()
        tmp["cluster"] = tmp["winter_cluster"]
        plot_map_cluster(tmp, path)

    def _plot_map_price_winter(df_local: pd.DataFrame, path: str):
        plot_map_price(df_local, path)

    # Heatmap cluster × winter_feature
    def _plot_heatmap_cluster_vs_winter(df_local: pd.DataFrame, path: str):
        labels_loc = np.asarray(df_local["winter_cluster"])
        clusters = sorted(c for c in np.unique(labels_loc) if c >= 0)
        if not clusters:
            return
        shap_cols = winter_cols
        col_labels = FEATURE_NAMES
        M = np.zeros((len(clusters), len(shap_cols)))
        for i, c in enumerate(clusters):
            mask = labels_loc == c
            M[i] = df_local.loc[mask, shap_cols].mean(axis=0).values
        fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(col_labels)), max(4, 0.4 * len(clusters))))
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
        plt.colorbar(im, ax=ax, label="Moyenne Winter-like")
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, ha="right", fontsize=8)
        ax.set_title("Cluster × feature (moyenne Winter-like)")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    _plot_umap_price_winter(df, os.path.join(out_dir, "winter_umap_price.png"))
    _plot_umap_cluster_winter(df, os.path.join(out_dir, "winter_umap_cluster.png"))
    _plot_map_cluster_winter(df, os.path.join(out_dir, "winter_map_cluster.png"))
    _plot_map_price_winter(df, os.path.join(out_dir, "winter_map_price.png"))
    _plot_heatmap_cluster_vs_winter(df, os.path.join(out_dir, "winter_heatmap_cluster_vs_feature.png"))

    print("Figures Winter-like enregistrées dans", out_dir)


if __name__ == "__main__":
    main()

