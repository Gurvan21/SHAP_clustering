import argparse, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.explain.order1 import Order1TreeSHAP, Order1PermutationSHAP
from mosaic_shap.reduction.pca import PCAReducer
from mosaic_shap.reduction.umap_red import UMAPReducer
from mosaic_shap.clustering.hdbscan_clusterer import HDBSCANClusterer
from mosaic_shap.pipeline.summarize import summarize_order1_by_cluster
from mosaic_shap.vis.plots import scatter_2d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["rf","lr"], default="rf")
    ap.add_argument("--algo", choices=["tree","perm"], default="tree")
    ap.add_argument("--pca_dim", type=int, default=20)
    ap.add_argument("--min_cluster_size", type=int, default=60)
    ap.add_argument("--min_samples", type=int, default=None)
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    X, y, meta = make_dataset_overlap_scores_but_separable_interactions(n=args.n, seed=args.seed, p_noise=4)
    fn = meta["feature_names"]

    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=250, random_state=args.seed).fit(X, y)
    else:
        model = LogisticRegression(max_iter=800).fit(X, y)

    if args.algo == "tree":
        phi1 = Order1TreeSHAP(check_additivity=False).compute(model, X)
    else:
        rng = np.random.default_rng(args.seed)
        bg = X[rng.choice(len(X), size=min(200, len(X)), replace=False)]
        phi1 = Order1PermutationSHAP(background=bg, max_evals=1200).compute(model, X[:300])
        X = X[:300]; y = y[:300]  # keep consistent

    # reduce for clustering/vis
    Z = phi1
    #Zp = PCAReducer(n_components=min(args.pca_dim, Z.shape[1]), random_state=args.seed).fit_transform(Z)
    Z2d = UMAPReducer(random_state=args.seed).fit_transform(Z)

    labels = HDBSCANClusterer(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples).fit_predict(Z2d)

    scatter_2d(Z2d, labels, "UMAP on SHAP order-1 (colored by cluster)", "figures/order1_umap_by_cluster.png")
    scatter_2d(Z2d, y, "UMAP on SHAP order-1 (colored by class)", "figures/order1_umap_by_class.png")

    df = summarize_order1_by_cluster(phi1, labels, fn, topk=8)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
