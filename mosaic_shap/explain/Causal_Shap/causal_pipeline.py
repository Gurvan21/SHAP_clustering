"""
Pipeline d'exploration causale : SCM (DoWhy GCM), graphe NetworkX, do-Shapley, clustering, heatmaps.
Inspiré du notebook DoShapley_pipeline et du do-calcul (interventions).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from dowhy import gcm


def build_default_synthetic_graph():
    """DAG type notebook : Z1,Z2 -> X1,X2,X3 -> Y."""
    G = nx.DiGraph()
    G.add_edges_from([
        ("Z1", "X1"), ("Z1", "X2"), ("Z1", "X3"),
        ("Z2", "X1"), ("Z2", "X2"), ("Z2", "X3"),
        ("X1", "Y"), ("X2", "Y"), ("X3", "Y"),
    ])
    return G


def build_california_graph(feature_names: list, target_name: str = "price") -> nx.DiGraph:
    """DAG California Housing : toutes les features -> target (régression prix). Utilisé seulement si pas de découverte."""
    G = nx.DiGraph()
    for f in feature_names:
        G.add_edge(f, target_name)
    return G


def discover_causal_dag(df: pd.DataFrame, method: str = "ges", **kwargs) -> nx.DiGraph:
    """
    Exploration causale : découvre les liens causaux entre les variables à partir des données.
    method : 'ges' (Greedy Equivalence Search, BIC) ou 'pc' (tests d'indépendance conditionnelle).
    Retourne un DAG NetworkX avec les noms des colonnes de df comme nœuds.
    """
    from .causal_discovery import discover_dag
    return discover_dag(df, method=method, **kwargs)


def draw_and_save_graph(G: nx.DiGraph, path: str, target_name: str | None = None):
    """Dessine le DAG avec NetworkX et sauvegarde en PNG. Layout basé sur la structure du graphe (comme dans le GraphML)."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    nodes = list(G.nodes)
    # Utiliser un layout qui reflète les arêtes du graphe (spring = force-directed) pour que la figure corresponde au GraphML
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_size=2200, arrowsize=20,
        node_color="#E0E0E0", edgecolors="black", linewidths=1.5, font_size=11, ax=ax,
    )
    ax.set_title("DAG causal (SCM)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fit_scm(G: nx.DiGraph, df: pd.DataFrame) -> Any:
    """Construit et ajuste le SCM DoWhy GCM."""
    scm = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(scm, df)
    gcm.fit(scm, df)
    return scm


def estimate_v_do(scm, x_row: pd.Series, S: list, target: str, n_mc: int = 400) -> float:
    """E[Y | do(X_S = x_S)] par échantillonnage interventionnel."""
    interventions = {name: (lambda v, c=float(x_row[name]): c) for name in S}
    samples = gcm.interventional_samples(scm, interventions=interventions, num_samples_to_draw=n_mc)
    return float(np.mean(samples[target].to_numpy()))


def do_shapley_for_one(scm, x_row: pd.Series, features: list, target: str, n_perm: int = 60, n_mc: int = 250, seed: int = 0) -> dict:
    """Valeurs do-Shapley pour un individu (approximation par permutations)."""
    rng = np.random.default_rng(seed)
    phi = {f: 0.0 for f in features}
    cache = {}

    def v(S_list):
        key = tuple(sorted(S_list))
        if key not in cache:
            cache[key] = estimate_v_do(scm, x_row, list(key), target, n_mc=n_mc)
        return cache[key]

    for _ in range(n_perm):
        perm = rng.permutation(features)
        S = []
        v_prev = v(S)
        for f in perm:
            S.append(f)
            v_new = v(S)
            phi[f] += (v_new - v_prev)
            v_prev = v_new

    for f in features:
        phi[f] /= n_perm
    return phi


def do_shapley_matrix(scm, X_df: pd.DataFrame, features: list, target: str, n_perm: int = 60, n_mc: int = 250, seed: int = 0) -> np.ndarray:
    """Matrice (n, n_features) des do-Shapley."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(X_df), len(features)))
    for i in range(len(X_df)):
        phi_i = do_shapley_for_one(scm, X_df.iloc[i], features=features, target=target, n_perm=n_perm, n_mc=n_mc, seed=int(rng.integers(0, 10_000_000)))
        out[i] = [phi_i[f] for f in features]
    return out


def run_causal_pipeline(
    df: pd.DataFrame,
    G: nx.DiGraph | None = None,
    feature_names: list | None = None,
    target_name: str = "Y",
    out_dir: str | None = None,
    n_perm: int = 60,
    n_mc: int = 250,
    seed: int = 0,
    min_cluster_size: int = 50,
    min_samples: int = 20,
) -> dict[str, Any]:
    """
    Pipeline complet : fit SCM, graphe, do-Shapley, UMAP, HDBSCAN, heatmap, scatter.
    df doit contenir les colonnes du graphe (dont target_name).
    feature_names = liste des nœuds sur lesquels on calcule do-Shapley (ex. X1,X2,X3,Z1,Z2).
    """
    from mosaic_shap.reduction import create_reducer
    from mosaic_shap.clustering import create_clusterer

    if G is None:
        G = build_default_synthetic_graph()
    if feature_names is None:
        feature_names = [n for n in G.nodes if n != target_name]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        draw_and_save_graph(G, os.path.join(out_dir, "causal_graph.png"), target_name=target_name)
        nx.write_graphml(G, os.path.join(out_dir, "causal_graph.graphml"))

    scm = fit_scm(G, df)
    X_df = df[feature_names].copy()
    y = df[target_name].to_numpy()

    do_phi = do_shapley_matrix(scm, X_df, feature_names, target_name, n_perm=n_perm, n_mc=n_mc, seed=seed)

    reducer = create_reducer("umap", n_components=2, random_state=seed)
    Z_red = reducer.fit_transform(do_phi)
    labels = create_clusterer("hdbscan", min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(Z_red)

    result = {
        "do_shapley": do_phi,
        "Z_reduced": Z_red,
        "labels": labels,
        "y": y,
        "feature_names": feature_names,
        "scm": scm,
        "G": G,
    }

    if out_dir:
        out_df = pd.DataFrame(do_phi, columns=[f"do_shap_{f}" for f in feature_names])
        out_df["target"] = y
        out_df["umap_1"] = Z_red[:, 0]
        out_df["umap_2"] = Z_red[:, 1]
        out_df["cluster"] = labels
        if "Latitude" in df.columns and "Longitude" in df.columns:
            out_df["Latitude"] = df["Latitude"].values
            out_df["Longitude"] = df["Longitude"].values
        out_df.to_csv(os.path.join(out_dir, "do_shapley_clusters.csv"), index=False)
        result["csv_path"] = os.path.join(out_dir, "do_shapley_clusters.csv")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(Z_red[:, 0], Z_red[:, 1], c=labels, s=10, alpha=0.7)
        ax.set_title("UMAP sur do-Shapley – couleur = cluster")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "umap_cluster.png"), dpi=180)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(Z_red[:, 0], Z_red[:, 1], c=y, s=10, alpha=0.7, cmap="viridis")
        plt.colorbar(sc, ax=ax, label=target_name)
        ax.set_title("UMAP sur do-Shapley – couleur = cible")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "umap_target.png"), dpi=180)
        plt.close()

        clusters = sorted(c for c in np.unique(labels) if c >= 0)
        if clusters:
            M = np.zeros((len(clusters), len(feature_names)))
            for i, c in enumerate(clusters):
                mask = labels == c
                M[i] = do_phi[mask].mean(axis=0)
            fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(feature_names)), max(4, 0.3 * len(clusters))))
            v = np.abs(M).max()
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            plt.colorbar(im, ax=ax, label="Moyenne do-Shapley")
            ax.set_yticks(range(len(clusters)))
            ax.set_yticklabels([f"Cluster {int(c)}" for c in clusters])
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, ha="right")
            ax.set_title("Cluster × feature (moyenne do-Shapley)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "heatmap_cluster_vs_doshapley.png"), dpi=180)
            plt.close()

        if "Latitude" in out_df.columns and "Longitude" in out_df.columns:
            _plot_causal_maps(out_df, out_dir, target_name)

    return result


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
    try:
        import contextily as cx
        x, y = _to_mercator(lon, lat)
        if x is not None:
            cx.add_basemap(ax, crs="EPSG:3857", zoom=6, alpha=0.8)
    except Exception:
        pass


def _plot_causal_maps(out_df: pd.DataFrame, out_dir: str, target_name: str):
    """Carte Californie (lat/lon) : couleur = cluster puis couleur = cible (prix)."""
    lon = out_df["Longitude"].values
    lat = out_df["Latitude"].values
    x_merc, y_merc = _to_mercator(lon, lat)
    use_merc = x_merc is not None

    # Carte par cluster
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = out_df["cluster"].values
    u = np.unique(labels)
    u = np.concatenate([u[u >= 0], u[u < 0]])
    for c in u:
        mask = labels == c
        if use_merc:
            ax.scatter(x_merc[mask], y_merc[mask], s=5, alpha=0.6, label=f"Cluster {int(c)}")
        else:
            ax.scatter(lon[mask], lat[mask], s=5, alpha=0.6, label=f"Cluster {int(c)}")
    ax.set_xlabel("Longitude" + (" (Mercator)" if use_merc else ""))
    ax.set_ylabel("Latitude" + (" (Mercator)" if use_merc else ""))
    ax.set_title("Carte Californie (do-Shapley) – couleur = cluster")
    if use_merc:
        ax.set_xlim(x_merc.min() - 1e4, x_merc.max() + 1e4)
        ax.set_ylim(y_merc.min() - 1e4, y_merc.max() + 1e4)
        _add_basemap_if_available(ax, lon, lat)
    else:
        ax.set_aspect("equal")
        ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
        ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "map_cluster.png"), dpi=180)
    plt.close()

    # Carte par prix (target)
    fig, ax = plt.subplots(figsize=(8, 8))
    if use_merc:
        ax.set_xlim(x_merc.min() - 1e4, x_merc.max() + 1e4)
        ax.set_ylim(y_merc.min() - 1e4, y_merc.max() + 1e4)
        sc = ax.scatter(x_merc, y_merc, c=out_df["target"], s=5, cmap="viridis", alpha=0.6)
    else:
        sc = ax.scatter(lon, lat, c=out_df["target"], s=5, cmap="viridis", alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
        ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    plt.colorbar(sc, ax=ax, label=target_name + " (×100k $)" if target_name == "price" else target_name)
    ax.set_xlabel("Longitude" + (" (Mercator)" if use_merc else ""))
    ax.set_ylabel("Latitude" + (" (Mercator)" if use_merc else ""))
    ax.set_title("Carte Californie (do-Shapley) – couleur = " + target_name)
    if use_merc:
        _add_basemap_if_available(ax, lon, lat)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "map_price.png"), dpi=180)
    plt.close()
