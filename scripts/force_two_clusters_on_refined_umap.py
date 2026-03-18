#!/usr/bin/env python3
"""
Forcer un clustering en K clusters évidents à partir de l'UMAP déjà calculé
dans figures/california_xgb_joint_refined/joint_shap_refined.csv, sans
modifier les données d'origine (SHAP, interactions, etc.).

Stratégie :
  - Charger le CSV refined (avec umap_1_refined, umap_2_refined),
  - Appliquer un KMeans à K clusters dans l'espace UMAP,
  - Sauvegarder un nouveau CSV + figures (UMAP, cartes, heatmap) avec K clusters.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _plot_umap(df: pd.DataFrame, color_col: str, path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_col == "price":
        sc = ax.scatter(df["umap_1_refined"], df["umap_2_refined"], c=df["price"], s=8, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    else:
        labels = df[color_col].values
        u = np.unique(labels)
        for c in u:
            m = labels == c
            ax.scatter(df.loc[m, "umap_1_refined"], df.loc[m, "umap_2_refined"], s=8, alpha=0.7, label=f"Cluster {int(c)}")
        ax.legend(markerscale=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_map(df: pd.DataFrame, color_col: str, path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_col == "price":
        sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["price"], s=8, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    else:
        labels = df[color_col].values
        u = np.unique(labels)
        for c in u:
            m = labels == c
            ax.scatter(df.loc[m, "Longitude"], df.loc[m, "Latitude"], s=8, alpha=0.7, label=f"Cluster {int(c)}")
        ax.legend(markerscale=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_heatmap(df: pd.DataFrame, label_col: str, comp_cols, path: str):
    labels = np.asarray(df[label_col])
    clusters = sorted(c for c in np.unique(labels) if c >= 0)
    if not clusters:
        return
    M = np.zeros((len(clusters), len(comp_cols)))
    for i, c in enumerate(clusters):
        m = labels == c
        M[i] = df.loc[m, comp_cols].mean(axis=0).values
    vmax = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(comp_cols)), max(4, 0.45 * len(clusters))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Moyenne composante")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.set_xticks(range(len(comp_cols)))
    ax.set_xticklabels(comp_cols, rotation=90, ha="right", fontsize=8)
    ax.set_title(f"{len(clusters)} clusters × composantes (refined)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Forcer K clusters (KMeans) sur l'UMAP raffiné, sans modifier les données.")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    K = int(args.k)

    base = _base_dir()
    in_csv = os.path.join(base, "figures/california_xgb_joint_refined/joint_shap_refined.csv")
    out_dir = os.path.join(base, f"figures/california_xgb_joint_{K}clusters")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "umap_1_refined" not in df.columns or "umap_2_refined" not in df.columns:
        raise ValueError("Les colonnes umap_1_refined / umap_2_refined sont manquantes.")

    U = df[["umap_1_refined", "umap_2_refined"]].values

    # KMeans en K clusters sur l'UMAP
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    labels = kmeans.fit_predict(U)

    df_out = df.copy()
    label_col = f"cluster_{K}"
    df_out[label_col] = labels

    df_out.to_csv(os.path.join(out_dir, f"joint_shap_{K}clusters.csv"), index=False)

    # Colonnes de composantes pour la heatmap : on utilise les joint_* (même base que refined)
    comp_cols = [c for c in df.columns if c.startswith("joint_")]

    _plot_umap(df_out, label_col, os.path.join(out_dir, "umap_cluster.png"),
               title=f"UMAP (refined) – {K} clusters forcés (KMeans)")
    _plot_umap(df_out, "price", os.path.join(out_dir, "umap_price.png"),
               title="UMAP (refined) – couleur = prix")
    _plot_map(df_out, label_col, os.path.join(out_dir, "map_cluster.png"),
              title=f"Carte Californie – {K} clusters (KMeans sur UMAP)")
    _plot_map(df_out, "price", os.path.join(out_dir, "map_price.png"),
              title="Carte Californie – prix")
    _plot_heatmap(df_out, label_col, comp_cols, os.path.join(out_dir, "heatmap_cluster_vs_components.png"))

    print(f"Terminé. Résultats ({K} clusters) dans", out_dir)


if __name__ == "__main__":
    main()

