"""
Tests génériques qui comparent toutes les méthodes d'explicabilité
sur les mêmes données et vérifient leur cohérence mutuelle.
"""
import numpy as np
import pytest
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.explain.Classical_Shap.order1 import Order1TreeSHAP
from mosaic_shap.explain.Classical_Shap.order2 import Order2TreeSHAPInteractions
from mosaic_shap.explain.Owen_Shap.owen_explainer import OWENExplainer
from mosaic_shap.explain.Winter_Shap.winter_explainer import WINTERExplainer

@pytest.fixture(scope="module")
def shared_setup():
    X, y, _ = make_dataset_overlap_scores_but_separable_interactions(n=300, seed=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1).fit(X, y)
    M = X.shape[1]
    background = X[:80]
    groups = [list(range(M//2)), list(range(M//2, M))]
    fn = [f"x{i}" for i in range(M)]
    shap1  = Order1TreeSHAP().compute(model, X[:100]).values
    shap2  = Order2TreeSHAPInteractions().compute(model, X[:100]).values
    owen   = OWENExplainer(model, groups, background, 32, fn).shap_values(X[:50])
    winter = WINTERExplainer(model, [list(range(M))], groups, background, 32, fn).shap_values(X[:50])
    return {"X": X, "model": model, "M": M, "background": background,
            "groups": groups, "fn": fn,
            "shap1": shap1, "shap2": shap2, "owen": owen, "winter": winter}


def test_all_efficiency(shared_setup):
    """Toutes les méthodes respectent l'axiome d'efficience."""
    d = shared_setup
    baseline = d["model"].predict(d["background"]).mean()
    N = 50

    for name, vals in [("Shapley", d["shap1"][:N]),
                        ("Owen",    d["owen"][:N]),
                        ("Winter",  d["winter"][:N])]:
        preds = d["model"].predict(d["X"][:N])
        err   = np.abs(vals.sum(1) - (preds - baseline)).mean()
        assert err < 0.1, f"{name} : erreur d'efficience {err:.4f}"


def test_ranking_consistency(shared_setup):
    """Shapley, Owen et Winter doivent donner des rankings corrélés (τ > 0.4)."""
    d = shared_setup
    N = min(len(d["shap1"]), len(d["owen"]), len(d["winter"]))
    imp_shap   = np.abs(d["shap1"][:N]).mean(0)
    imp_owen   = np.abs(d["owen"][:N]).mean(0)
    imp_winter = np.abs(d["winter"][:N]).mean(0)

    tau_so, _ = kendalltau(imp_shap, imp_owen)
    tau_sw, _ = kendalltau(imp_shap, imp_winter)
    tau_ow, _ = kendalltau(imp_owen, imp_winter)

    assert tau_so > 0.4, f"Shapley vs Owen : τ={tau_so:.3f} trop faible"
    assert tau_sw > 0.4, f"Shapley vs Winter : τ={tau_sw:.3f} trop faible"
    assert tau_ow > 0.6, f"Owen vs Winter : τ={tau_ow:.3f} trop faible"


def test_order2_diagonal_is_order1(shared_setup):
    """La diagonale de shap2 doit être cohérente avec shap1."""
    d = shared_setup
    N = min(len(d["shap1"]), len(d["shap2"]))
    diag = np.array([d["shap2"][n, :, :].diagonal() for n in range(N)])
    corr = np.corrcoef(np.abs(d["shap1"][:N]).mean(0),
                       np.abs(diag).mean(0))[0, 1]
    assert corr > 0.5, f"Diagonale ordre 2 vs ordre 1 : r={corr:.3f}"


def test_non_negativity_of_importance(shared_setup):
    """Les importances moyennes (mean |φ|) sont toujours positives."""
    d = shared_setup
    for name, vals in [("Shapley", d["shap1"]),
                        ("Owen",    d["owen"]),
                        ("Winter",  d["winter"])]:
        imp = np.abs(vals).mean(0)
        assert (imp >= 0).all(), f"{name} : importance négative détectée"