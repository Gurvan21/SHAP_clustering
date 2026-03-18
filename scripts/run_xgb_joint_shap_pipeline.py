#!/usr/bin/env python3
"""
California Housing (dataset complet) + XGBoost + SHAP ordre 1 & ordre 2 simultanément.

Objectif :
  - entraîner un XGBoost (régression) sur tous les points du dataset California Housing,
  - calculer SHAP interaction values (ordre 2) sur tous les points,
  - définir SHAP ordre 1 "pur" = diagonale des interactions (main effects),
    afin que l'ordre 1 ne contienne plus les traces d'ordre 2,
  - sélectionner les top-k effets principaux et top-k interactions par importance RMS,
  - vectoriser ces composantes sélectionnées, projeter UMAP, clusteriser (HDBSCAN),
  - produire les figures (UMAP cluster/prix, carte cluster/prix, heatmap clusters × composantes)
    et sauvegarder un CSV réutilisable.

Sortie :
  figures/california_xgb_joint/
    - joint_shap.csv
    - umap_cluster.png
    - umap_price.png
    - map_cluster.png
    - map_price.png
    - heatmap_cluster_vs_joint_shap.png
    - selected_components.txt
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.data.Data_Pedagogique import make_dataset_Housing_California
from mosaic_shap.reduction import create_reducer
from mosaic_shap.clustering import create_clusterer


FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def _base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fit_xgb_regressor(X: np.ndarray, y: np.ndarray, seed: int):
    from xgboost import XGBRegressor

    # Paramètres raisonnables (CPU) – n_jobs=-1 pour utiliser la machine
    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    return model.fit(X, y)


def _shap_interactions(model, X: np.ndarray) -> np.ndarray:
    import shap

    expl = shap.TreeExplainer(model)
    inter = expl.shap_interaction_values(X)
    if isinstance(inter, list):
        # régression => souvent liste de taille 1
        inter = inter[0]
    inter = np.asarray(inter)
    # attend (n, p, p)
    if inter.ndim == 4:
        inter = inter[:, :, :, 0]
    return inter


def _rms(a: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(a), axis=axis))


@dataclass(frozen=True)
class Component:
    kind: str  # "main" ou "inter"
    i: int
    j: int  # pour main: i==j, pour inter: i<j
    name: str


def _select_components(
    inter: np.ndarray,
    top_k_main: int,
    top_k_inter: int,
) -> Tuple[List[Component], np.ndarray]:
    """
    inter: (n, p, p)
    Retourne une liste de composantes sélectionnées + matrice vectorisée V (n, d).
    """
    n, p, p2 = inter.shape
    assert p == p2

    main = np.stack([inter[:, i, i] for i in range(p)], axis=1)  # (n, p)
    main_scores = _rms(main, axis=0)
    main_order = np.argsort(main_scores)[::-1]
    sel_main = main_order[: min(top_k_main, p)]

    # Interactions : triangle supérieur i<j
    pairs = []
    pair_scores = []
    for i in range(p):
        for j in range(i + 1, p):
            vals = inter[:, i, j]
            pairs.append((i, j))
            pair_scores.append(float(_rms(vals, axis=0)))
    pair_scores = np.asarray(pair_scores)
    pair_order = np.argsort(pair_scores)[::-1]
    sel_pairs = [pairs[idx] for idx in pair_order[: min(top_k_inter, len(pairs))]]

    comps: List[Component] = []
    cols: List[np.ndarray] = []

    for i in sel_main:
        name = FEATURE_NAMES[i]
        comps.append(Component(kind="main", i=int(i), j=int(i), name=f"main:{name}"))
        cols.append(main[:, i])

    for (i, j) in sel_pairs:
        name = f"{FEATURE_NAMES[i]}×{FEATURE_NAMES[j]}"
        comps.append(Component(kind="inter", i=int(i), j=int(j), name=f"inter:{name}"))
        cols.append(inter[:, i, j])

    V = np.stack(cols, axis=1) if cols else np.zeros((n, 0))
    return comps, V


def _standardize(V: np.ndarray) -> np.ndarray:
    if V.size == 0:
        return V
    return (V - V.mean(axis=0, keepdims=True)) / (V.std(axis=0, keepdims=True) + 1e-8)


def plot_umap_price(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["umap_1"], df["umap_2"], c=df["price"], s=8, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    ax.set_title("UMAP (joint SHAP) – couleur = prix")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_umap_cluster(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = df["cluster"].values
    u = np.unique(labels)
    u = np.concatenate([u[u >= 0], u[u < 0]])
    for c in u:
        mask = labels == c
        ax.scatter(df.loc[mask, "umap_1"], df.loc[mask, "umap_2"], s=8, alpha=0.7, label=f"Cluster {int(c)}")
    ax.set_title("UMAP (joint SHAP) – couleur = cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_map_cluster(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = df["cluster"].values
    u = np.unique(labels)
    u = np.concatenate([u[u >= 0], u[u < 0]])
    for c in u:
        mask = labels == c
        ax.scatter(df.loc[mask, "Longitude"], df.loc[mask, "Latitude"], s=8, alpha=0.7, label=f"Cluster {int(c)}")
    ax.set_title("Carte Californie – couleur = cluster (joint SHAP)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_map_price(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["price"], s=8, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    ax.set_title("Carte Californie – couleur = prix")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap_cluster_vs_components(df: pd.DataFrame, comp_cols: List[str], path: str):
    labels = np.asarray(df["cluster"])
    clusters = sorted(c for c in np.unique(labels) if c >= 0)
    if not clusters:
        return
    M = np.zeros((len(clusters), len(comp_cols)))
    for i, c in enumerate(clusters):
        mask = labels == c
        M[i] = df.loc[mask, comp_cols].mean(axis=0).values
    vmax = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(comp_cols)), max(4, 0.45 * len(clusters))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Moyenne (joint SHAP composante)")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.set_xticks(range(len(comp_cols)))
    ax.set_xticklabels(comp_cols, rotation=90, ha="right", fontsize=8)
    ax.set_title("Clusters × composantes (main + interactions sélectionnées)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Pipeline joint SHAP (ordre1 main + ordre2 interactions) sur XGBoost (California Housing).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=20640, help="Nombre de points (20640 ~ dataset complet).")
    parser.add_argument("--top-k-main", type=int, default=8, help="Top-k effets principaux (main effects).")
    parser.add_argument("--top-k-inter", type=int, default=12, help="Top-k interactions (i<j).")
    parser.add_argument("--min-cluster-size", type=int, default=80)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--out", type=str, default="figures/california_xgb_joint")
    args = parser.parse_args()

    base = _base_dir()
    out_dir = os.path.join(base, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Dataset complet via n=20640 (échantillonnage sans remplacement)
    X, y, meta = make_dataset_Housing_California(n=args.n, seed_=args.seed)
    fn = list(meta["feature_names"])
    if fn != FEATURE_NAMES:
        # On se cale sur l'ordre du dataset (mais on garde FEATURE_NAMES pour noms attendus)
        pass

    model = _fit_xgb_regressor(X, y, seed=args.seed)

    # SHAP interactions sur tous les points
    inter = _shap_interactions(model, X)  # (n, p, p)
    if inter.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"SHAP interactions: p={inter.shape[1]} inattendu.")

    comps, V = _select_components(inter, top_k_main=args.top_k_main, top_k_inter=args.top_k_inter)
    Vn = _standardize(V)

    Z_red = create_reducer("umap", n_components=2, random_state=args.seed).fit_transform(Vn)
    labels = create_clusterer("hdbscan", min_cluster_size=args.min_cluster_size, min_samples=args.min_samples).fit_predict(Z_red)

    # DataFrame de sortie
    df = pd.DataFrame(X, columns=[f"feat_{c}" for c in FEATURE_NAMES])
    df["price"] = y
    df["Latitude"] = X[:, FEATURE_NAMES.index("Latitude")]
    df["Longitude"] = X[:, FEATURE_NAMES.index("Longitude")]

    # Stocker main effects "purs" (diagonale)
    main = np.stack([inter[:, i, i] for i in range(len(FEATURE_NAMES))], axis=1)
    for i, name in enumerate(FEATURE_NAMES):
        df[f"main_{name}"] = main[:, i]

    # Stocker interactions complètes (triangle sup) seulement pour les sélectionnées (pour le clustering/heatmap)
    comp_cols: List[str] = []
    for k, comp in enumerate(comps):
        col = f"joint_{k:02d}_{comp.name.replace(':', '_')}"
        df[col] = V[:, k]
        comp_cols.append(col)

    df["umap_1"] = Z_red[:, 0]
    df["umap_2"] = Z_red[:, 1]
    df["cluster"] = labels

    csv_path = os.path.join(out_dir, "joint_shap.csv")
    df.to_csv(csv_path, index=False)

    with open(os.path.join(out_dir, "selected_components.txt"), "w") as f:
        f.write("Selected components for clustering (in order):\n")
        for k, comp in enumerate(comps):
            f.write(f"{k:02d}  {comp.name}\n")

    # Figures
    plot_umap_cluster(df, os.path.join(out_dir, "umap_cluster.png"))
    plot_umap_price(df, os.path.join(out_dir, "umap_price.png"))
    plot_map_cluster(df, os.path.join(out_dir, "map_cluster.png"))
    plot_map_price(df, os.path.join(out_dir, "map_price.png"))
    plot_heatmap_cluster_vs_components(df, comp_cols, os.path.join(out_dir, "heatmap_cluster_vs_joint_shap.png"))

    print("Terminé.")
    print("CSV:", csv_path)
    print("Figures:", out_dir)


if __name__ == "__main__":
    main()

