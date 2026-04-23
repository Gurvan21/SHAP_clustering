import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.explain.Classical_Shap.order2 import Order2TreeSHAPInteractions
from mosaic_shap.pipeline.vectorize import vectorize_interactions
from mosaic_shap.reduction.pca import PCAReducer
from mosaic_shap.clustering.hdbscan_clusterer import HDBSCANClusterer

def test_order2_pipeline_runs():
    X, y, _ = make_dataset_overlap_scores_but_separable_interactions(n=240, seed=0, p_noise=0)
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(X, y)
    shap2 = Order2TreeSHAPInteractions().compute(model, X[:80]).values
    Z2, iu = vectorize_interactions(shap2, include_diag=False)
    Zp = PCAReducer(n_components=min(10, Z2.shape[1]), random_state=0).fit_transform(Z2)
    labels = HDBSCANClusterer(min_cluster_size=15).fit_predict(Zp)
    assert labels.shape[0] == Zp.shape[0]

def test_order2_interpretability():
    """Les top interactions par cluster doivent être des paires valides."""
    from mosaic_shap.pipeline.interpret import top_interactions_per_cluster

    X, y, _ = make_dataset_overlap_scores_but_separable_interactions(n=240, seed=0)
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(X, y)
    shap2  = Order2TreeSHAPInteractions().compute(model, X[:80]).values
    labels = np.array([0]*40 + [1]*40)
    M = X.shape[1]
    feature_names = [f"x{i}" for i in range(M)]

    top = top_interactions_per_cluster(shap2, labels, feature_names, top_k=3)
    assert set(top.keys()) == {0, 1}
    for pairs in top.values():
        assert len(pairs) <= 3


def test_estimator_consistency():
    """Tree et MonteCarlo doivent donner des rankings corrélés (τ > 0.5)."""
    from mosaic_shap.explain.order2 import Order2MonteCarloInteractions
    from scipy.stats import kendalltau

    X, y, _ = make_dataset_overlap_scores_but_separable_interactions(n=100, seed=0)
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)

    tree_vals = Order2TreeSHAPInteractions().compute(model, X[:30]).values
    mc_vals   = Order2MonteCarloInteractions(n_perms=40).compute(model, X[:30]).values

    # Importance par paire : mean |φᵢⱼ|
    tree_imp = np.abs(tree_vals).mean(0).ravel()
    mc_imp   = np.abs(mc_vals).mean(0).ravel()

    tau, _ = kendalltau(tree_imp, mc_imp)
    assert tau > 0.5, f"Rankings Tree vs MonteCarlo trop divergents : τ={tau:.3f}"