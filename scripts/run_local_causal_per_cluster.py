#!/usr/bin/env python3
"""
Exploration causale locale par cluster (ordre 1 California).
Pour chaque cluster, on ne garde que les top-k features (les plus importantes en |SHAP| moyen),
puis on lance la découverte causale (GES) sur ce sous-ensemble de variables + price.
Résultats: figures/causal_shap_california_per_order1_cluster/cluster_0/, cluster_1/, ...
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.explain.Causal_Shap.causal_pipeline import (
    discover_causal_dag,
    draw_and_save_graph,
)
import networkx as nx

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_NAME = "price"


def top_features_per_cluster(df: pd.DataFrame, cluster_id: int, top_k: int = 5) -> list:
    """Retourne les noms des top-k features pour le cluster (par moyenne |SHAP|)."""
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    if not shap_cols:
        return FEATURE_NAMES[: top_k]
    sub = df.loc[df["cluster"] == cluster_id, shap_cols]
    means = sub.abs().mean()
    # noms sans préfixe shap_
    names = [c.replace("shap_", "") for c in means.index]
    order = np.argsort(means.values)[::-1]
    return [names[i] for i in order[: top_k]]


def run_local_causal_for_cluster(
    df_full: pd.DataFrame,
    cluster_id: int,
    out_dir: str,
    top_k: int = 5,
    discovery_method: str = "ges",
):
    """
    Pour un cluster donné: extrait les points du cluster, garde les top-k features + price,
    lance la découverte causale (GES/PC), sauvegarde le DAG local.
    """
    mask = df_full["cluster"] == cluster_id
    df_c = df_full.loc[mask].copy()
    if len(df_c) < 30:
        print(f"  cluster_{cluster_id}: trop peu de points ({len(df_c)}), ignoré.")
        return

    top_feat = top_features_per_cluster(df_full, cluster_id, top_k=top_k)
    # Colonnes feat_* dans le CSV
    feat_cols = [f"feat_{f}" for f in top_feat]
    local_df = df_c[feat_cols + ["price"]].copy()
    local_df.columns = top_feat + ["price"]

    G = discover_causal_dag(local_df, method=discovery_method)
    os.makedirs(out_dir, exist_ok=True)
    draw_and_save_graph(G, os.path.join(out_dir, "causal_graph.png"), target_name=TARGET_NAME)
    nx.write_graphml(G, os.path.join(out_dir, "causal_graph.graphml"))

    # Résumé: features utilisées, nombre d'arêtes
    with open(os.path.join(out_dir, "local_features.txt"), "w") as f:
        f.write("top_features: " + ", ".join(top_feat) + "\n")
        f.write("n_points: " + str(len(df_c)) + "\n")
        f.write("n_edges: " + str(G.number_of_edges()) + "\n")

    print(f"  cluster_{cluster_id}: n={len(df_c)}, top features={top_feat}, arêtes={G.number_of_edges()} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Exploration causale locale par cluster ordre 1 (top-k features).")
    parser.add_argument("--csv", type=str, default=None, help="CSV ordre 1 (défaut: figures/california_order1/shap_clusters.csv)")
    parser.add_argument("--out", type=str, default="figures/causal_shap_california_per_order1_cluster", help="Répertoire de sortie")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de features les plus importantes par cluster")
    parser.add_argument("--discovery", type=str, default="ges", choices=["ges", "pc"])
    parser.add_argument("--clusters", type=str, default=None, help="Clusters à traiter, ex. 0,1,2 ou tous si absent")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = args.csv or os.path.join(base, "figures/california_order1/shap_clusters.csv")
    out_parent = os.path.join(base, args.out)

    if not os.path.isfile(csv_path):
        print("Fichier introuvable:", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "cluster" not in df.columns:
        print("Le CSV doit contenir une colonne 'cluster'.")
        sys.exit(1)

    clusters = sorted(c for c in np.unique(df["cluster"].values) if c >= 0)
    if args.clusters:
        clusters = [int(x) for x in args.clusters.split(",")]
    print(f"Exploration causale locale pour {len(clusters)} clusters (top-{args.top_k} features), méthode {args.discovery}")
    for c in clusters:
        subdir = os.path.join(out_parent, f"cluster_{int(c)}")
        run_local_causal_for_cluster(
            df, c, subdir, top_k=args.top_k, discovery_method=args.discovery
        )
    print("Terminé.")


if __name__ == "__main__":
    main()
