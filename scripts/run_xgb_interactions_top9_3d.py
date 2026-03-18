#!/usr/bin/env python3
"""
Clustering en 3D UMAP avec les 9 interactions les plus fortes, puis clustering,
et visualisations adaptées (projections 2D).

On repart de :
  figures/california_xgb_joint/joint_shap.csv

Étapes :
  - sélectionner les 9 colonnes joint_* d'interactions les plus fortes (RMS),
  - standardiser,
  - UMAP en 3 dimensions,
  - KMeans (par défaut 4 clusters) dans l'espace UMAP 3D,
  - visualiser en 2D via les 3 paires (UMAP1,UMAP2), (UMAP1,UMAP3), (UMAP2,UMAP3)
    couleur = cluster et couleur = prix,
  - cartes + heatmap.

Sorties :
  figures/california_xgb_joint_3d_inter9/
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.reduction import create_reducer


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _rms_cols(X: np.ndarray) -> np.ndarray:
    return np.sqrt((X ** 2).mean(axis=0))


def _standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)


def _scatter_2d(df, xcol, ycol, color_col, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_col == "price":
        sc = ax.scatter(df[xcol], df[ycol], c=df["price"], s=8, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    else:
        labels = df[color_col].values
        u = np.unique(labels)
        for c in u:
            m = labels == c
            ax.scatter(df.loc[m, xcol], df.loc[m, ycol], s=8, alpha=0.7, label=f"Cluster {int(c)}")
        ax.legend(markerscale=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
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
    clusters = sorted(np.unique(labels))
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
    ax.set_title(f"{len(clusters)} clusters × 9 interactions (3D UMAP)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    base = _base_dir()
    in_csv = os.path.join(base, "figures/california_xgb_joint/joint_shap.csv")
    out_dir = os.path.join(base, "figures/california_xgb_joint_3d_inter9")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv)
    inter_cols = [c for c in df.columns if c.startswith("joint_") and "inter_" in c]
    if len(inter_cols) < 9:
        raise ValueError(f"Moins de 9 colonnes d'interactions joint_* trouvées ({len(inter_cols)}).")

    X = df[inter_cols].values
    rms = _rms_cols(X)
    order = np.argsort(rms)[::-1]
    top9_idx = order[:9]
    top9_cols = [inter_cols[i] for i in top9_idx]

    X9 = X[:, top9_idx]
    X9s = _standardize(X9)

    # UMAP 3D
    reducer = create_reducer("umap", n_components=3, random_state=0, n_neighbors=40, min_dist=0.1)
    U3 = reducer.fit_transform(X9s)  # shape (n,3)

    # KMeans à 4 clusters dans l'espace UMAP 3D
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
    labels = kmeans.fit_predict(U3)

    df_out = df.copy()
    df_out["umap_1_3d"] = U3[:, 0]
    df_out["umap_2_3d"] = U3[:, 1]
    df_out["umap_3_3d"] = U3[:, 2]
    df_out["cluster_inter9_4_3d"] = labels

    df_out.to_csv(os.path.join(out_dir, "joint_shap_4clusters_inter9_3d.csv"), index=False)

    with open(os.path.join(out_dir, "top9_interactions.txt"), "w") as f:
        f.write("Top 9 joint_* interactions by RMS (3D run):\n")
        for c in top9_cols:
            f.write(c + "\n")

    # Visualisations 2D des trois paires
    _scatter_2d(
        df_out,
        "umap_1_3d",
        "umap_2_3d",
        "cluster_inter9_4_3d",
        os.path.join(out_dir, "umap12_cluster.png"),
        "UMAP (dim1,dim2) – 4 clusters (interactions top9, 3D)",
    )
    _scatter_2d(
        df_out,
        "umap_1_3d",
        "umap_2_3d",
        "price",
        os.path.join(out_dir, "umap12_price.png"),
        "UMAP (dim1,dim2) – couleur = prix",
    )
    _scatter_2d(
        df_out,
        "umap_1_3d",
        "umap_3_3d",
        "cluster_inter9_4_3d",
        os.path.join(out_dir, "umap13_cluster.png"),
        "UMAP (dim1,dim3) – 4 clusters (interactions top9, 3D)",
    )
    _scatter_2d(
        df_out,
        "umap_1_3d",
        "umap_3_3d",
        "price",
        os.path.join(out_dir, "umap13_price.png"),
        "UMAP (dim1,dim3) – couleur = prix",
    )
    _scatter_2d(
        df_out,
        "umap_2_3d",
        "umap_3_3d",
        "cluster_inter9_4_3d",
        os.path.join(out_dir, "umap23_cluster.png"),
        "UMAP (dim2,dim3) – 4 clusters (interactions top9, 3D)",
    )
    _scatter_2d(
        df_out,
        "umap_2_3d",
        "umap_3_3d",
        "price",
        os.path.join(out_dir, "umap23_price.png"),
        "UMAP (dim2,dim3) – couleur = prix",
    )

    # Carte et heatmap
    _plot_map(
        df_out,
        "cluster_inter9_4_3d",
        os.path.join(out_dir, "map_cluster.png"),
        "Carte Californie – 4 clusters (UMAP 3D interactions top9)",
    )
    _plot_map(
        df_out,
        "price",
        os.path.join(out_dir, "map_price.png"),
        "Carte Californie – prix",
    )
    _plot_heatmap(
        df_out,
        "cluster_inter9_4_3d",
        top9_cols,
        os.path.join(out_dir, "heatmap_cluster_vs_interactions.png"),
    )

    print("Terminé. Résultats (UMAP 3D + 4 clusters avec 9 interactions) dans", out_dir)


if __name__ == "__main__":
    main()

