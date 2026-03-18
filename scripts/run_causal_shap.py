#!/usr/bin/env python3
"""
Pipeline causal : exploration (SCM DoWhy GCM), graphe NetworkX (affichage + stockage),
do-Shapley, puis UMAP, clustering HDBSCAN, heatmap cluster × feature, scatter.
Utilisation : données synthétiques (défaut) ou CSV avec colonnes alignées au graphe.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.data.Data_Pedagogique import make_dataset_Housing_California
from mosaic_shap.explain.Causal_Shap.causal_pipeline import (
    build_california_graph,
    build_default_synthetic_graph,
    discover_causal_dag,
    run_causal_pipeline,
)


def make_synthetic_causal_data(n: int = 800, seed: int = 0):
    """Données synthétiques compatibles avec le DAG Z1,Z2 -> X1,X2,X3 -> Y."""
    rng = np.random.default_rng(seed)
    Z1 = rng.uniform(-3, 3, size=n)
    Z2 = rng.uniform(-3, 3, size=n)
    e = 0.3 * rng.normal(size=(3, n))
    X1 = 0.9 * Z1 + 0.3 * Z2 + e[0]
    X2 = -0.9 * Z1 + 0.2 * Z2 + e[1]
    X3 = 0.7 * Z1 + 0.1 * Z2 + e[2]
    score = 2.0 * (X1 - X2) + 0.5 * X3 + 0.5 * rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -10, 10)))
    Y = rng.binomial(1, prob)
    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "Z1": Z1, "Z2": Z2, "Y": Y})
    return df


CALIFORNIA_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def main():
    parser = argparse.ArgumentParser(description="Pipeline causal: SCM, graphe, do-Shapley, cluster, heatmap.")
    parser.add_argument("--out-dir", type=str, default=None, help="Répertoire de sortie (défaut: figures/causal_shap ou figures/causal_shap_california)")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "california"], help="synthetic ou california (3000 points)")
    parser.add_argument("--csv", type=str, default=None, help="CSV avec colonnes alignées au graphe; ignoré si --dataset california")
    parser.add_argument("--n", type=int, default=500, help="Taille échantillon (synthétique uniquement)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-perm", type=int, default=40, help="Permutations pour do-Shapley")
    parser.add_argument("--n-mc", type=int, default=200, help="Tirages Monte Carlo par coalition")
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--n-subset", type=int, default=None, help="Pour california: sous-échantillon des 3000 points (même seed = mêmes points); défaut 3000")
    parser.add_argument("--discovery", type=str, default="ges", choices=["ges", "pc", "none"], help="Exploration causale: découvrir le DAG depuis les données (ges, pc) ou 'none' pour graphe fixe.")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.dataset == "california":
        X, y, meta = make_dataset_Housing_California(n=3000, seed_=args.seed)
        fn = list(meta["feature_names"])
        df = pd.DataFrame(X, columns=fn)
        df["price"] = y
        if args.n_subset is not None and args.n_subset < len(df):
            df = df.iloc[: args.n_subset].copy()
        target_name = "price"
        if args.discovery and args.discovery != "none":
            print("Exploration causale: découverte du DAG avec", args.discovery.upper(), "...")
            G = discover_causal_dag(df, method=args.discovery)
            feature_names = [n for n in G.nodes if n != target_name]
            print("DAG découvert: %d nœuds, %d arêtes" % (G.number_of_nodes(), G.number_of_edges()))
        else:
            G = build_california_graph(fn, target_name=target_name)
            feature_names = fn
        out_dir = args.out_dir or "figures/causal_shap_california"
    else:
        if args.csv:
            df = pd.read_csv(args.csv)
            required = ["X1", "X2", "X3", "Z1", "Z2", "Y"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise SystemExit(f"CSV doit contenir les colonnes: {required}. Manquantes: {missing}")
            df = df[required].copy()
        else:
            df = make_synthetic_causal_data(n=args.n, seed=args.seed)
        G = build_default_synthetic_graph()
        feature_names = ["X1", "X2", "X3", "Z1", "Z2"]
        target_name = "Y"
        out_dir = args.out_dir or "figures/causal_shap"
    out_dir = os.path.join(base, out_dir)

    result = run_causal_pipeline(
        df,
        G=G,
        feature_names=feature_names,
        target_name=target_name,
        out_dir=out_dir,
        n_perm=args.n_perm,
        n_mc=args.n_mc,
        seed=args.seed,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    print("Graphe sauvegardé:", os.path.join(out_dir, "causal_graph.png"))
    print("Graphe (GraphML):", os.path.join(out_dir, "causal_graph.graphml"))
    print("CSV do-Shapley + clusters:", result.get("csv_path", out_dir))
    print("Clusters (do-Shapley):", sorted(c for c in np.unique(result["labels"]) if c >= 0))


if __name__ == "__main__":
    main()
