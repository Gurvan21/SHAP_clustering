"""
Single configurable pipeline: data -> model -> SHAP (order 1 or 2) -> reduce -> cluster -> summarize/plot.
Config keys: dataset, model, order, explain, reducer, cluster, n, seed, out_dir, max_explain_samples, etc.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..clustering import create_clusterer
from ..explain import Order1PermutationSHAP, Order1TreeSHAP, Order2Interactions
from ..reduction import create_reducer
from .vectorize import vectorize_interactions
from .summarize import summarize_order1_by_cluster, summarize_interactions_by_cluster


def load_data(dataset: str, n: int, seed: int, **kwargs) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load (X, y, meta) with meta['feature_names']."""
    if dataset == "synthetic":
        from ..data.synthetic import make_dataset_overlap_scores_but_separable_interactions
        p_noise = kwargs.get("p_noise", 4)
        return make_dataset_overlap_scores_but_separable_interactions(n=n, seed=seed, p_noise=p_noise)
    if dataset == "housing":
        from ..data.Data_Pedagogique import make_dataset_Housing_California
        return make_dataset_Housing_California(n=n, seed_=seed)
    raise ValueError(f"Unknown dataset: {dataset}. Use 'synthetic' or 'housing'.")


def fit_model(model_name: str, X: np.ndarray, y: np.ndarray, seed: int):
    """Return fitted model."""
    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=250, random_state=seed).fit(X, y)
    if model_name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=800).fit(X, y)
    if model_name == "xgb":
        from sklearn import ensemble
        return ensemble.GradientBoostingRegressor(
            n_estimators=500, max_depth=4, min_samples_split=5, learning_rate=0.01
        ).fit(X, y)
    raise ValueError(f"Unknown model: {model_name}. Use 'rf', 'lr', 'xgb'.")


def run(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run the full pipeline. Returns dict with keys: X, y, model, feature_names,
    Z_reduced, labels, df_summary, (phi1, Z2, iu for order2), etc.
    """
    n = config.get("n", 1200)
    seed = config.get("seed", 0)
    dataset = config.get("dataset", "synthetic")
    model_name = config.get("model", "rf")
    order = config.get("order", 1)
    explain = config.get("explain", "tree")
    if explain == "perm":
        explain = "permutation"
    elif explain == "mc":
        explain = "montecarlo"
    elif explain == "reg":
        explain = "regression"
    reducer = config.get("reducer", "umap")
    cluster = config.get("cluster", "hdbscan")
    out_dir = config.get("out_dir")
    max_explain_samples = config.get("max_explain_samples")
    show_plots = config.get("show_plots", False)

    data_kw = {k: config[k] for k in ("p_noise",) if k in config}
    X, y, meta = load_data(dataset, n=n, seed=seed, **data_kw)
    fn = meta["feature_names"]
    model = fit_model(model_name, X, y, seed)

    if order == 1:
        if explain == "tree":
            phi1 = Order1TreeSHAP(check_additivity=False).compute(model, X)
        else:
            rng = np.random.default_rng(seed)
            n_bg = min(200, len(X))
            bg = X[rng.choice(len(X), size=n_bg, replace=False)]
            phi1 = Order1PermutationSHAP(background=bg, max_evals=config.get("max_evals", 1200)).compute(model, X)
        Z = phi1
        iu = None
        Z2 = None
        X_use, y_use = X, y
    else:
        X_use = X if max_explain_samples is None else X[:max_explain_samples]
        y_use = y if max_explain_samples is None else y[:max_explain_samples]
        if explain == "tree":
            shap2 = Order2Interactions(method="tree").compute(model, X_use).values
        elif explain == "montecarlo":
            shap2 = Order2Interactions(
                method="montecarlo", budget=config.get("budget", 1200), random_state=seed
            ).compute(model, X_use).values
        else:
            shap2 = Order2Interactions(
                method="regression", budget=config.get("budget", 350), random_state=seed
            ).compute(model, X_use).values
        Z2, iu = vectorize_interactions(shap2, include_diag=False)
        Z = Z2
        phi1 = None

    Z_reduced = create_reducer(
        reducer,
        n_components=config.get("n_components", 2),
        random_state=seed,
    ).fit_transform(Z)

    cluster_kw = {}
    if cluster == "hdbscan":
        cluster_kw = {
            "min_cluster_size": config.get("min_cluster_size", 60),
            "min_samples": config.get("min_samples"),
        }
    elif cluster in ("kmeans", "agglomerative"):
        cluster_kw = {"n_clusters": config.get("n_clusters", 8)}
    labels = create_clusterer(cluster, random_state=seed, **cluster_kw).fit_predict(Z_reduced)

    result = {
        "X": X_use, "y": y_use, "model": model, "feature_names": fn,
        "Z_reduced": Z_reduced, "labels": labels,
        "order": order, "iu": iu,
    }

    if order == 1:
        result["phi1"] = phi1
        result["df_summary"] = summarize_order1_by_cluster(phi1, labels, fn, topk=config.get("topk", 8))
    else:
        result["Z2"] = Z2
        result["phi1"] = Order1TreeSHAP().compute(model, X_use) if model_name == "rf" else None
        result["df_interactions"] = summarize_interactions_by_cluster(Z2, labels, iu, fn, topk=config.get("topk", 8))
        result["df_summary"] = (
            summarize_order1_by_cluster(result["phi1"], labels, fn, topk=config.get("topk", 8))
            if result["phi1"] is not None else None
        )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        from ..vis.plots import scatter_2d, heatmap_clusters_vs_interactions
        prefix = "order1" if order == 1 else "order2"
        scatter_2d(Z_reduced, labels, f"SHAP order-{order} (cluster)", os.path.join(out_dir, f"{prefix}_umap_cluster.png"), show=show_plots)
        scatter_2d(Z_reduced, y_use, f"SHAP order-{order} (class)", os.path.join(out_dir, f"{prefix}_umap_class.png"), show=show_plots)
        if order == 2 and Z2 is not None:
            interaction_names = [f"{fn[i]}×{fn[j]}" for i, j in zip(iu[0], iu[1])]
            heatmap_clusters_vs_interactions(
                Z2, labels, interaction_names,
                topk=config.get("topk_heatmap", 25), path=os.path.join(out_dir, f"{prefix}_heatmap.png"), show=show_plots
            )
        if result.get("df_summary") is not None:
            result["df_summary"].to_csv(os.path.join(out_dir, f"{prefix}_summary_by_cluster.csv"), index=False)
        if result.get("df_interactions") is not None:
            result["df_interactions"].to_csv(os.path.join(out_dir, f"{prefix}_interactions_by_cluster.csv"), index=False)

    return result
