import numpy as np
from sklearn.datasets import fetch_california_housing



def make_dataset_Housing_California(n=1000, p_noise=4, seed=0, sigma=0.9):
    """Scores A/B overlap in score space, but regimes are separable in explainability space."""
    california_housing =fetch_california_housing()
    X, y = fetch_california_housing(return_X_y=True)

    meta = {
        "regime": fetch_california_housing,
        "feature_names": california_housing.feature_names,
    }
    return X, y, meta
