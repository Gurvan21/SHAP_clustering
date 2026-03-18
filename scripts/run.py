#!/usr/bin/env python3
"""Single entry point: SHAP clustering pipeline. All options via CLI."""
import argparse
import os

from mosaic_shap.pipeline.run import run


def main():
    p = argparse.ArgumentParser(description="SHAP order-1/2 -> reduce -> cluster -> summarize & plots")
    p.add_argument("--dataset", choices=["synthetic", "housing"], default="synthetic")
    p.add_argument("--model", choices=["rf", "lr", "xgb"], default="rf")
    p.add_argument("--order", type=int, choices=[1, 2], default=1)
    p.add_argument("--explain", default="tree",
                    help="order1: tree|perm; order2: tree|montecarlo|regression")
    p.add_argument("--reducer", choices=["pca", "umap"], default="umap")
    p.add_argument("--cluster", choices=["hdbscan", "kmeans", "agglomerative"], default="hdbscan")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="figures", help="Output directory for plots (default: figures)")
    p.add_argument("--show-plots", action="store_true", help="Display plots (default: save only)")
    p.add_argument("--max-explain-samples", type=int, default=None,
                    help="Max samples for order-2 explain (default: all)")
    p.add_argument("--min-cluster-size", type=int, default=60)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--n-clusters", type=int, default=8, help="For kmeans/agglomerative")
    p.add_argument("--budget", type=int, default=1200, help="For order-2 montecarlo/regression")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--topk-heatmap", type=int, default=25)
    p.add_argument("--p-noise", type=int, default=4, help="Synthetic dataset only")
    args = p.parse_args()

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "order": args.order,
        "explain": args.explain,
        "reducer": args.reducer,
        "cluster": args.cluster,
        "n": args.n,
        "seed": args.seed,
        "out_dir": (args.out_dir or "figures").strip() or "figures",
        "show_plots": args.show_plots,
        "max_explain_samples": args.max_explain_samples,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "n_clusters": args.n_clusters,
        "budget": args.budget,
        "topk": args.topk,
        "topk_heatmap": args.topk_heatmap,
        "p_noise": args.p_noise,
    }
    result = run(config)

    if result.get("df_summary") is not None:
        print("\nOrder-1 summary by cluster:")
        print(result["df_summary"].to_string(index=False))
    if result.get("df_interactions") is not None:
        print("\nTop interactions per cluster:")
        print(result["df_interactions"].to_string(index=False))


if __name__ == "__main__":
    main()
