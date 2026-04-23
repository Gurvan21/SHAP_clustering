import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.explain.Owen_Shap.owen_explainer import OWENExplainer
from mosaic_shap.explain.Winter_Shap.winter_explainer import WINTERExplainer

@pytest.fixture(scope="module")
def setup():
    X, y, _ = make_dataset_overlap_scores_but_separable_interactions(n=200, seed=0)
    model = RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=1).fit(X, y)
    background = X[:50]
    M = X.shape[1]
    groups = [list(range(M//2)), list(range(M//2, M))]
    return X, y, model, background, groups, M


def test_owen_efficiency(setup):
    """Σ φᵢ_Owen = f(x) - E[f(X)]"""
    X, y, model, background, groups, M = setup
    explainer = OWENExplainer(model, groups, background, n_permutations=32,
                               feature_names=[f"x{i}" for i in range(M)])
    N = 20
    ov = explainer.shap_values(X[:N])
    preds    = model.predict(X[:N])
    baseline = model.predict(background).mean()
    err = np.abs(ov.sum(axis=1) - (preds - baseline)).mean()
    assert err < 0.05, f"Erreur d'efficience Owen trop grande : {err:.4f}"


def test_owen_shape(setup):
    """Output shape = (N, M)"""
    X, y, model, background, groups, M = setup
    explainer = OWENExplainer(model, groups, background, n_permutations=16,
                               feature_names=[f"x{i}" for i in range(M)])
    ov = explainer.shap_values(X[:10])
    assert ov.shape == (10, M)


def test_winter_efficiency(setup):
    """Σ φᵢ_Winter = f(x) - E[f(X)]"""
    X, y, model, background, groups, M = setup
    coarse = [list(range(M))]
    fine   = groups
    explainer = WINTERExplainer(model, coarse, fine, background, n_permutations=32,
                                 feature_names=[f"x{i}" for i in range(M)])
    N = 20
    wv = explainer.shap_values(X[:N])
    preds    = model.predict(X[:N])
    baseline = model.predict(background).mean()
    err = np.abs(wv.sum(axis=1) - (preds - baseline)).mean()
    assert err < 0.05, f"Erreur d'efficience Winter trop grande : {err:.4f}"


def test_winter_flat_equals_owen(setup):
    """Winter avec hiérarchie plate ≈ Owen (corrélation > 0.90)"""
    X, y, model, background, groups, M = setup
    N = 15
    feature_names = [f"x{i}" for i in range(M)]

    owen_exp = OWENExplainer(model, groups, background, n_permutations=32,
                              feature_names=feature_names)
    ov = owen_exp.shap_values(X[:N])

    winter_exp = WINTERExplainer(model, groups, groups, background,
                                  n_permutations=32, feature_names=feature_names)
    wv = winter_exp.shap_values(X[:N])

    corr = np.corrcoef(ov.ravel(), wv.ravel())[0, 1]
    assert corr > 0.90, f"Corrélation Winter(plate) vs Owen trop faible : {corr:.3f}"


def test_winter_shape(setup):
    """Output shape = (N, M)"""
    X, y, model, background, groups, M = setup
    coarse = [list(range(M))]
    explainer = WINTERExplainer(model, coarse, groups, background, n_permutations=16,
                                 feature_names=[f"x{i}" for i in range(M)])
    wv = explainer.shap_values(X[:10])
    assert wv.shape == (10, M)


def test_owen_null_player(setup):
    """Une feature constante doit avoir φ_Owen ≈ 0"""
    X, y, model, background, groups, M = setup
    X_const = X.copy()
    X_const[:, 0] = X_const[:, 0].mean()   # feature 0 constante
    explainer = OWENExplainer(model, groups, background, n_permutations=64,
                               feature_names=[f"x{i}" for i in range(M)])
    ov = explainer.shap_values(X_const[:20])
    assert np.abs(ov[:, 0]).mean() < 0.1, "Feature constante devrait avoir φ ≈ 0"