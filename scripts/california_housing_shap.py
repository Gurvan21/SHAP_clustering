#!/usr/bin/env python3
"""
California Housing: SHAP ordre 1 ou 2 (régression) sur N points, clustering, CSV réutilisable.
4 vues: UMAP (prix / cluster), carte CA (cluster / prix). Mêmes points et même modèle pour ordre 1 et 2.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.data.Data_Pedagogique import make_dataset_Housing_California
from mosaic_shap.explain import Order1TreeSHAP, Order2Interactions
from mosaic_shap.explain.tree_winter_like import WinterLikeTreeAttributor
from mosaic_shap.pipeline.vectorize import vectorize_interactions
from mosaic_shap.reduction import create_reducer
from mosaic_shap.clustering import create_clusterer

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
SUBDIR_ORDER1 = "figures/california_order1"
SUBDIR_ORDER2 = "figures/california_order2"
SUBDIR_ORDER2_PER_O1 = "figures/california_order2_per_order1_cluster"
CSV_NAME = "shap_clusters.csv"
FEAT_PREFIX = "feat_"


def _base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fit_model(X, y, seed: int):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=4, min_samples_split=5, learning_rate=0.05, random_state=seed
    ).fit(X, y)


def load_or_compute(seed: int = 0, n: int = 3000, order: int = 1, force_recompute: bool = False):
    """Charge le CSV si présent, sinon calcule SHAP (ordre 1 ou 2) + clusters et sauvegarde. Mêmes points et même modèle."""
    subdir = SUBDIR_ORDER1 if order == 1 else SUBDIR_ORDER2
    path = os.path.join(_base_dir(), subdir, CSV_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not force_recompute and os.path.isfile(path):
        df = pd.read_csv(path)
        return df, False

    X, y, meta = make_dataset_Housing_California(n=n, seed_=seed)
    fn = list(meta["feature_names"])
    model = _fit_model(X, y, seed)

    if order == 1:
        phi = Order1TreeSHAP().compute(model, X)
        phi = np.asarray(phi)
        Z = phi
    else:
        # GradientBoostingRegressor non supporté par shapiq TreeExplainer → on utilise shap
        import shap
        expl = shap.TreeExplainer(model)
        shap2 = expl.shap_interaction_values(X)
        if isinstance(shap2, list):
            shap2 = shap2[0]
        shap2 = np.asarray(shap2)
        if shap2.ndim == 4:
            shap2 = shap2[:, :, :, 0]
        Z, _ = vectorize_interactions(shap2, include_diag=False)

    Z_red = create_reducer("umap", n_components=2, random_state=seed).fit_transform(Z)
    labels = create_clusterer("hdbscan", min_cluster_size=80, min_samples=20).fit_predict(Z_red)

    cols_feat = [f"feat_{f}" for f in fn]
    df = pd.DataFrame(X, columns=cols_feat)
    df["price"] = y
    df["Latitude"] = X[:, fn.index("Latitude")]
    df["Longitude"] = X[:, fn.index("Longitude")]

    if order == 1:
        for i, name in enumerate(fn):
            df[f"shap_{name}"] = phi[:, i]
        # Ajout des contributions Winter-like par feature (approximation hiérarchique locale)
        winter = WinterLikeTreeAttributor(model, FEATURE_NAMES)
        phi_w, baseline_w, preds_w = winter.explain_batch(X)
        max_err = winter.max_efficiency_error(X)
        print(f"[Winter-like] max efficiency error on batch: {max_err:.3e}")
        for i, name in enumerate(FEATURE_NAMES):
            df[f"winter_{name}"] = phi_w[:, i]
    else:
        for j in range(Z.shape[1]):
            df[f"shap2_{j}"] = Z[:, j]

    df["umap_1"] = Z_red[:, 0]
    df["umap_2"] = Z_red[:, 1]
    df["cluster"] = labels

    df.to_csv(path, index=False)
    return df, True


def _compute_order2_on_subset(X_sub: np.ndarray, y_sub: np.ndarray, model, fn: list, seed: int):
    """Calcule SHAP ordre 2 + UMAP + clustering sur un sous-ensemble. Retourne un DataFrame comme load_or_compute order=2."""
    import shap
    expl = shap.TreeExplainer(model)
    shap2 = expl.shap_interaction_values(X_sub)
    if isinstance(shap2, list):
        shap2 = shap2[0]
    shap2 = np.asarray(shap2)
    if shap2.ndim == 4:
        shap2 = shap2[:, :, :, 0]
    Z, _ = vectorize_interactions(shap2, include_diag=False)
    n_sub = Z.shape[0]
    min_size = max(10, min(50, n_sub // 3))
    Z_red = create_reducer("umap", n_components=2, random_state=seed).fit_transform(Z)
    labels = create_clusterer("hdbscan", min_cluster_size=min_size, min_samples=max(5, min_size // 2)).fit_predict(Z_red)
    cols_feat = [f"{FEAT_PREFIX}{f}" for f in fn]
    df = pd.DataFrame(X_sub, columns=cols_feat)
    df["price"] = y_sub
    df["Latitude"] = X_sub[:, fn.index("Latitude")]
    df["Longitude"] = X_sub[:, fn.index("Longitude")]
    for j in range(Z.shape[1]):
        df[f"shap2_{j}"] = Z[:, j]
    df["umap_1"] = Z_red[:, 0]
    df["umap_2"] = Z_red[:, 1]
    df["cluster"] = labels
    return df


def run_order2_per_order1_cluster(seed: int = 0, force_recompute: bool = False):
    """
    Pour chaque cluster d'ordre 1, lance SHAP ordre 2 + clustering sur ses points.
    Sauvegarde dans figures/california_order2_per_order1_cluster/cluster_0/, cluster_1/, ...
    """
    base = _base_dir()
    o1_csv = os.path.join(base, SUBDIR_ORDER1, CSV_NAME)
    if not os.path.isfile(o1_csv):
        raise FileNotFoundError(f"Ordre 1 requis: exécutez d'abord --order 1. Fichier attendu: {o1_csv}")
    df_o1 = pd.read_csv(o1_csv)
    cluster_col = df_o1["cluster"]
    feat_cols = [c for c in df_o1.columns if c.startswith(FEAT_PREFIX)]
    fn = [c.replace(FEAT_PREFIX, "") for c in feat_cols]
    X = df_o1[feat_cols].values
    y = df_o1["price"].values
    model = _fit_model(X, y, seed)
    clusters = sorted(c for c in np.unique(cluster_col) if c >= 0)
    out_parent = os.path.join(base, SUBDIR_ORDER2_PER_O1)
    os.makedirs(out_parent, exist_ok=True)
    for c in clusters:
        subdir = os.path.join(out_parent, f"cluster_{int(c)}")
        csv_path = os.path.join(subdir, CSV_NAME)
        idx = (np.asarray(cluster_col) == c)
        X_sub = X[idx]
        y_sub = y[idx]
        if len(X_sub) < 20:
            continue
        os.makedirs(subdir, exist_ok=True)
        if not force_recompute and os.path.isfile(csv_path):
            df_sub = pd.read_csv(csv_path)
        else:
            df_sub = _compute_order2_on_subset(X_sub, y_sub, model, fn, seed)
            df_sub.to_csv(csv_path, index=False)
        plot_umap_price(df_sub, os.path.join(subdir, "umap_price.png"), order=2)
        plot_umap_cluster(df_sub, os.path.join(subdir, "umap_cluster.png"), order=2)
        plot_map_cluster(df_sub, os.path.join(subdir, "map_cluster.png"))
        plot_map_price(df_sub, os.path.join(subdir, "map_price.png"))
        plot_heatmap_cluster_vs_shap(df_sub, os.path.join(subdir, "heatmap_cluster_vs_shap.png"), order=2, feature_names=fn)
        print(f"  cluster_{int(c)}: {len(df_sub)} points -> {subdir}")
    print(f"Résultats dans {out_parent}")


def _order2_interaction_labels(p: int, fn: list):
    """Labels pour les colonnes shap2_0..shap2_d (triangle sup. sans diag)."""
    iu = np.triu_indices(p, k=1)
    return [f"{fn[iu[0][j]]}×{fn[iu[1][j]]}" for j in range(iu[0].size)]


def plot_heatmap_cluster_vs_shap(df: pd.DataFrame, path: str, order: int = 1, feature_names: list = None):
    """
    Heatmap: lignes = clusters, colonnes = features (ordre 1) ou paires d'interactions (ordre 2).
    Valeur = moyenne des valeurs SHAP dans le cluster.
    """
    labels = np.asarray(df["cluster"])
    clusters = sorted(c for c in np.unique(labels) if c >= 0)
    if not clusters:
        return
    if order == 1:
        shap_cols = [c for c in df.columns if c.startswith("shap_") and not c.startswith("shap2_")]
        col_labels = [c.replace("shap_", "") for c in shap_cols]
    else:
        shap_cols = [c for c in df.columns if c.startswith("shap2_")]
        p = 8
        fn = feature_names or FEATURE_NAMES
        col_labels = _order2_interaction_labels(p, fn)
        if len(col_labels) != len(shap_cols):
            col_labels = [f"I{j}" for j in range(len(shap_cols))]
    M = np.zeros((len(clusters), len(shap_cols)))
    for i, c in enumerate(clusters):
        mask = labels == c
        M[i] = df.loc[mask, shap_cols].mean(axis=0).values
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(col_labels)), max(4, 0.4 * len(clusters))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    plt.colorbar(im, ax=ax, label="Moyenne SHAP")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, ha="right", fontsize=8)
    ax.set_title(f"Cluster × {'feature' if order == 1 else 'interaction'} (moyenne SHAP ordre {order})")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_umap_price(df: pd.DataFrame, path: str, order: int = 1):
    """UMAP avec couleur = prix (heat)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["umap_1"], df["umap_2"],
        c=df["price"], s=8, cmap="viridis", alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    ax.set_title(f"Vue UMAP (ordre {order}) – couleur = prix (faible → fort)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_umap_cluster(df: pd.DataFrame, path: str, order: int = 1):
    """UMAP avec couleur = cluster."""
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = df["cluster"].values
    u = np.unique(labels)
    u = np.concatenate([u[u >= 0], u[u < 0]])
    for c in u:
        mask = labels == c
        ax.scatter(
            df.loc[mask, "umap_1"], df.loc[mask, "umap_2"],
            s=8, alpha=0.7, label=f"Cluster {int(c)}",
        )
    ax.set_title(f"Vue UMAP (ordre {order}) – couleur = cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _to_mercator(lon, lat):
    """Convertit lon/lat (WGS84) en Web Mercator pour fond de carte."""
    try:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = t.transform(lon, lat)
        return np.asarray(x), np.asarray(y)
    except Exception:
        return None, None


def _add_basemap_if_available(ax, lon, lat):
    """Ajoute une carte de fond si contextily + pyproj disponibles."""
    try:
        import contextily as cx
        x, y = _to_mercator(lon, lat)
        if x is not None:
            cx.add_basemap(ax, crs="EPSG:3857", zoom=6, alpha=0.8)
    except Exception:
        pass


def plot_map_cluster(df: pd.DataFrame, path: str):
    """Carte Californie (lon, lat) colorée par cluster."""
    fig, ax = plt.subplots(figsize=(8, 8))
    lon, lat = df["Longitude"].values, df["Latitude"].values
    x_merc, y_merc = _to_mercator(lon, lat)
    use_merc = x_merc is not None
    if use_merc:
        ax.set_xlim(x_merc.min() - 1e4, x_merc.max() + 1e4)
        ax.set_ylim(y_merc.min() - 1e4, y_merc.max() + 1e4)
    labels = df["cluster"].values
    u = np.unique(labels)
    u = np.concatenate([u[u >= 0], u[u < 0]])
    for c in u:
        mask = labels == c
        if use_merc:
            ax.scatter(x_merc[mask], y_merc[mask], s=5, alpha=0.6, label=f"Cluster {int(c)}")
        else:
            ax.scatter(lon[mask], lat[mask], s=5, alpha=0.6, label=f"Cluster {int(c)}")
    ax.set_xlabel("Longitude (ou X Mercator)" if use_merc else "Longitude")
    ax.set_ylabel("Latitude (ou Y Mercator)" if use_merc else "Latitude")
    ax.set_title("Carte Californie (ordre SHAP) – couleur = cluster")
    if not use_merc:
        ax.set_aspect("equal")
        ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
        ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    ax.legend(markerscale=2, fontsize=8)
    if use_merc:
        _add_basemap_if_available(ax, lon, lat)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_map_price(df: pd.DataFrame, path: str):
    """Carte Californie (lon, lat) colorée par prix."""
    fig, ax = plt.subplots(figsize=(8, 8))
    lon, lat = df["Longitude"].values, df["Latitude"].values
    x_merc, y_merc = _to_mercator(lon, lat)
    use_merc = x_merc is not None
    if use_merc:
        ax.set_xlim(x_merc.min() - 1e4, x_merc.max() + 1e4)
        ax.set_ylim(y_merc.min() - 1e4, y_merc.max() + 1e4)
        sc = ax.scatter(x_merc, y_merc, c=df["price"], s=5, cmap="viridis", alpha=0.6)
    else:
        sc = ax.scatter(lon, lat, c=df["price"], s=5, cmap="viridis", alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
        ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    plt.colorbar(sc, ax=ax, label="Prix (×100k $)")
    ax.set_xlabel("Longitude (ou X Mercator)" if use_merc else "Longitude")
    ax.set_ylabel("Latitude (ou Y Mercator)" if use_merc else "Latitude")
    ax.set_title("Carte Californie (ordre SHAP) – couleur = prix")
    if use_merc:
        _add_basemap_if_available(ax, lon, lat)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="California Housing: SHAP ordre 1 ou 2, clustering, 4 vues, CSV cache")
    p.add_argument("--order", type=int, choices=[1, 2], default=1, help="Ordre SHAP: 1 (valeurs) ou 2 (interactions)")
    p.add_argument("--order2-per-order1", action="store_true", help="Pour chaque cluster ordre 1, faire ordre 2 + clustering et stocker dans un sous-dossier")
    p.add_argument("--n", type=int, default=3000, help="Nombre de points (défaut 3000 pour ordre 1 et 2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Recalculer même si le CSV existe")
    p.add_argument("--no-figures", action="store_true", help="Ne pas générer les figures")
    args = p.parse_args()

    if args.order2_per_order1:
        run_order2_per_order1_cluster(seed=args.seed, force_recompute=args.force)
        return

    base = _base_dir()
    subdir = SUBDIR_ORDER1 if args.order == 1 else SUBDIR_ORDER2
    out_dir = os.path.join(base, subdir)
    os.makedirs(out_dir, exist_ok=True)
    csv_full = os.path.join(out_dir, CSV_NAME)

    df, computed = load_or_compute(seed=args.seed, n=args.n, order=args.order, force_recompute=args.force)
    print(f"Ordre {args.order} – Données: {len(df)} points. CSV: {'calculé et sauvegardé' if computed else 'chargé'} -> {csv_full}")

    if not args.no_figures:
        plot_umap_price(df, os.path.join(out_dir, "umap_price.png"), order=args.order)
        plot_umap_cluster(df, os.path.join(out_dir, "umap_cluster.png"), order=args.order)
        plot_map_cluster(df, os.path.join(out_dir, "map_cluster.png"))
        plot_map_price(df, os.path.join(out_dir, "map_price.png"))
        fn = [c.replace(FEAT_PREFIX, "") for c in df.columns if c.startswith(FEAT_PREFIX)]
        plot_heatmap_cluster_vs_shap(df, os.path.join(out_dir, "heatmap_cluster_vs_shap.png"), order=args.order, feature_names=fn)
        print(f"Figures enregistrées dans {out_dir}")


if __name__ == "__main__":
    main()
