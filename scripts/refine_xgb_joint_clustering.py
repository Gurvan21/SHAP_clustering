#!/usr/bin/env python3
"""
Affiner le clustering pour le pipeline XGBoost joint SHAP, sans recalculer SHAP.

On repart de figures/california_xgb_joint/joint_shap.csv (colonnes joint_*)
et on :
  - sélectionne un sous-ensemble de composantes (top-k ou seuil RMS),
  - standardise,
  - fait le clustering (HDBSCAN) dans l'espace "raw" (recommandé),
  - fait une projection UMAP uniquement pour la visualisation,
  - génère heatmap + UMAP + cartes.

Sorties dans figures/california_xgb_joint_refined/ (par défaut).
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.reduction import create_reducer
from mosaic_shap.clustering import create_clusterer


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)


def _rms_cols(X: np.ndarray) -> np.ndarray:
    return np.sqrt((X ** 2).mean(axis=0))


def _select_cols_main_plus_top_inter(
    cols: List[str],
    X: np.ndarray,
    n_top_inter: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Garde :
      - toutes les composantes main_* (effets principaux),
      - les n_top_inter composantes d'interaction les plus fortes (RMS) parmi les joint_* restantes.
    """
    rms = _rms_cols(X)
    order = np.argsort(rms)[::-1]

    main_cols: List[str] = []
    inter_candidates: List[Tuple[str, float]] = []
    for i in order:
        name = cols[i]
        if "main_" in name:
            main_cols.append(name)
        else:
            inter_candidates.append((name, rms[i]))

    # top n interactions
    inter_cols = [name for name, _ in inter_candidates[: max(0, n_top_inter)]]
    keep = main_cols + inter_cols
    X_keep = X[:, [cols.index(c) for c in keep]]
    return keep, X_keep


def _plot_umap(df: pd.DataFrame, xcol: str, ycol: str, color_col: str, path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_col == "price":
        sc = ax.scatter(df[xcol], df[ycol], c=df["price"], s=8, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    else:
        labels = df[color_col].values
        u = np.unique(labels)
        u = np.concatenate([u[u >= 0], u[u < 0]])
        for c in u:
            m = labels == c
            ax.scatter(df.loc[m, xcol], df.loc[m, ycol], s=8, alpha=0.7, label=f"Cluster {int(c)}")
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
        u = np.concatenate([u[u >= 0], u[u < 0]])
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


def _plot_heatmap(df: pd.DataFrame, cols: List[str], path: str):
    labels = np.asarray(df["cluster_refined"])
    clusters = sorted(c for c in np.unique(labels) if c >= 0)
    if not clusters:
        return
    M = np.zeros((len(clusters), len(cols)))
    for i, c in enumerate(clusters):
        m = labels == c
        M[i] = df.loc[m, cols].mean(axis=0).values
    vmax = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(cols)), max(4, 0.45 * len(clusters))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Moyenne composante")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, ha="right", fontsize=8)
    ax.set_title("Clusters × composantes sélectionnées (refined)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Affiner le clustering des composantes joint SHAP (sans recalcul SHAP).")
    parser.add_argument("--in-csv", type=str, default="figures/california_xgb_joint/joint_shap.csv")
    parser.add_argument("--out-dir", type=str, default="figures/california_xgb_joint_refined")
    parser.add_argument("--n-top-inter", type=int, default=2, help="Nombre d'interactions fortes à conserver en plus des main effects.")
    parser.add_argument("--umap-n-neighbors", type=int, default=40)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--hdb-min-cluster-size", type=int, default=80)
    parser.add_argument("--hdb-min-samples", type=int, default=20)
    args = parser.parse_args()

    base = _base_dir()
    in_csv = os.path.join(base, args.in_csv)
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv)
    joint_cols = [c for c in df.columns if c.startswith("joint_")]
    if not joint_cols:
        raise ValueError("Aucune colonne joint_* trouvée dans le CSV d'entrée.")

    X = df[joint_cols].values
    # Garde toutes les composantes main_* + n_top_inter plus fortes parmi les interactions
    keep_cols, X_keep = _select_cols_main_plus_top_inter(joint_cols, X, n_top_inter=args.n_top_inter)

    # Standardisation pour éviter que 2-3 composantes dominent
    Xs = _standardize(X_keep)

    # UMAP pour représentation
    reducer = create_reducer(
        "umap",
        n_components=2,
        random_state=0,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
    )
    U = reducer.fit_transform(Xs)

    # Clustering directement dans l'espace UMAP (ce que tu demandes)
    clusterer = create_clusterer(
        "hdbscan",
        min_cluster_size=args.hdb_min_cluster_size,
        min_samples=args.hdb_min_samples,
    )
    labels = clusterer.fit_predict(U)

    out = df.copy()
    out["cluster_refined"] = labels
    out["umap_1_refined"] = U[:, 0]
    out["umap_2_refined"] = U[:, 1]

    out.to_csv(os.path.join(out_dir, "joint_shap_refined.csv"), index=False)
    with open(os.path.join(out_dir, "kept_components.txt"), "w") as f:
        f.write("Kept joint_* components (sorted by RMS):\n")
        for c in keep_cols:
            f.write(c + "\n")

    _plot_umap(out, "umap_1_refined", "umap_2_refined", "cluster_refined", os.path.join(out_dir, "umap_cluster.png"),
               title="UMAP (refined) – couleur = cluster (clustering en espace raw)")
    _plot_umap(out, "umap_1_refined", "umap_2_refined", "price", os.path.join(out_dir, "umap_price.png"),
               title="UMAP (refined) – couleur = prix")
    _plot_map(out, "cluster_refined", os.path.join(out_dir, "map_cluster.png"),
              title="Carte Californie – clusters (refined)")
    _plot_map(out, "price", os.path.join(out_dir, "map_price.png"),
              title="Carte Californie – prix")
    _plot_heatmap(out, keep_cols, os.path.join(out_dir, "heatmap_cluster_vs_components.png"))

    print("Terminé. Sorties dans", out_dir)


if __name__ == "__main__":
    main()

