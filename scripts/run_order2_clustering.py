import argparse, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.explain.Classical_Shap.order2 import Order2TreeSHAPInteractions, Order2MonteCarloInteractions, Order2RegressionInteractions
from mosaic_shap.explain.Classical_Shap.order1 import Order1TreeSHAP, Order1PermutationSHAP
from mosaic_shap.pipeline.vectorize import vectorize_interactions
from mosaic_shap.pipeline.summarize import summarize_interactions_by_cluster, summarize_order1_by_cluster
from mosaic_shap.reduction.pca import PCAReducer
from mosaic_shap.reduction.umap_red import UMAPReducer
from mosaic_shap.clustering.hdbscan_clusterer import HDBSCANClusterer
from mosaic_shap.vis.plots import scatter_2d, heatmap_clusters_vs_interactions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["rf","lr"], default="rf")
    ap.add_argument("--algo", choices=["tree","mc","reg"], default="tree")
    ap.add_argument("--pca_dim", type=int, default=20)
    ap.add_argument("--min_cluster_size", type=int, default=60)
    ap.add_argument("--min_samples", type=int, default=None)
    ap.add_argument("--topk_heatmap", type=int, default=25)
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)

    X, y, meta = make_dataset_overlap_scores_but_separable_interactions(n=args.n, seed=args.seed, p_noise=2)
    fn = meta["feature_names"]

    # Train model
    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=250, random_state=args.seed).fit(X, y)
    else:
        model = LogisticRegression(max_iter=800).fit(X, y)

    # Compute order-2 interactions
    if args.algo == "tree":
        shap2 = Order2TreeSHAPInteractions().compute(model, X).values
    elif args.algo == "mc":
        Xs = X[:220]  # keep it tractable
        shap2 = Order2MonteCarloInteractions(n_perms=60, random_state=args.seed).compute(model, Xs).values
        X, y = Xs, y[:220]
    else:
        Xs = X[:220]
        shap2 = Order2RegressionInteractions(n_masks=350, random_state=args.seed).compute(model, Xs).values
        X, y = Xs, y[:220]

    Z2, iu = vectorize_interactions(shap2, include_diag=False)

    # PCA for clustering, UMAP for visualization
    #Zp = PCAReducer(n_components=min(args.pca_dim, Z2.shape[1]), random_state=args.seed).fit_transform(Z2)
    Z2d = UMAPReducer(random_state=args.seed).fit_transform(Z2)

    labels = HDBSCANClusterer(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples).fit_predict(Z2d)

    scatter_2d(Z2d, labels, "UMAP on order-2 interaction vectors (colored by cluster)", "figures/order2_umap_by_cluster.png")
    scatter_2d(Z2d, y, "UMAP on order-2 interaction vectors (colored by class)", "figures/order2_umap_by_class.png")

    # Build interaction names for heatmap
    interaction_names = [f"{fn[i]}×{fn[j]}" for i,j in zip(iu[0], iu[1])]
    heatmap_clusters_vs_interactions(Z2, labels, interaction_names, topk=args.topk_heatmap,title="Clusters vs interactions (mean signed)", path="figures/order2_heatmap_clusters_interactions.png")
    # Summaries
    df_int = summarize_interactions_by_cluster(Z2, labels, iu, fn, topk=8)
    print("\nTop interactions per cluster (mean, mean_abs):")
    print(df_int.to_string(index=False))

    # Interpret clusters using order-1 SHAP a posteriori (like your notebook)
    if args.model == "rf":
        phi1 = Order1TreeSHAP(check_additivity=False).compute(model, X)
    else:
        rng = np.random.default_rng(args.seed)
        bg = X[rng.choice(len(X), size=min(200, len(X)), replace=False)]
        phi1 = Order1PermutationSHAP(background=bg, max_evals=1200).compute(model, X)

    df_o1 = summarize_order1_by_cluster(phi1, labels, fn, topk=8)
    print("\nOrder-1 interpretation per cluster (top features):")
    print(df_o1.to_string(index=False))

if __name__ == "__main__":
    main()
