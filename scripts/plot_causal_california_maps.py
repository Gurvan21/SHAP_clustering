#!/usr/bin/env python3
"""Génère les cartes Californie (cluster, prix) à partir du CSV do-Shapley en rechargeant les coordonnées (même seed = même ordre)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mosaic_shap.data.Data_Pedagogique import make_dataset_Housing_California
from mosaic_shap.explain.Causal_Shap.causal_pipeline import _plot_causal_maps


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base, "figures/causal_shap_california/do_shapley_clusters.csv")
    out_dir = os.path.join(base, "figures/causal_shap_california")

    if not os.path.isfile(csv_path):
        print("Fichier introuvable:", csv_path)
        sys.exit(1)

    # Même 3000 points, seed=0 (ordre identique au pipeline)
    X, y, meta = make_dataset_Housing_California(n=3000, seed_=0)
    fn = list(meta["feature_names"])
    lat = X[:, fn.index("Latitude")]
    lon = X[:, fn.index("Longitude")]

    df = pd.read_csv(csv_path)
    if len(df) != len(lat):
        print("Attention: CSV a", len(df), "lignes, données California", len(lat))
    df["Latitude"] = lat[: len(df)]
    df["Longitude"] = lon[: len(df)]

    _plot_causal_maps(df, out_dir, "price")
    print("Cartes sauvegardées:", os.path.join(out_dir, "map_cluster.png"), os.path.join(out_dir, "map_price.png"))


if __name__ == "__main__":
    main()
