import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.explain.order2 import Order2TreeSHAPInteractions
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
