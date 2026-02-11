import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def make_dataset_Housing_California_Binary(n=1000, seed_=0, treshold=0.75):
    """Scores A/B overlap in score space, but regimes are separable in explainability space."""
    rng = np.random.default_rng(seed=seed_)
    california_housing = fetch_california_housing()
    df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    df['Target'] = california_housing.target

    df["Target"] = df["Target"]< (df.max()["Target"] * treshold)

    df[df["Target"]==False]["Target"] = 0
    df[df["Target"]==True]["Target"] = 1

    y = df["Target"].values
    X = df.drop(columns=["Target"]).values

    indices = rng.choice(len(y), size=n, replace=False)
    
    meta = {
        "regime": fetch_california_housing,
        "feature_names": california_housing.feature_names
    }
    return X[indices], y[indices], meta

def make_dataset_Housing_California(n=1000, seed_=0):
    """Scores A/B overlap in score space, but regimes are separable in explainability space."""
    rng = np.random.default_rng(seed=seed_)
    california_housing = fetch_california_housing()
    df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    df['Target'] = california_housing.target

    y = df["Target"].values
    X = df.drop(columns=["Target"]).values

    indices = rng.choice(len(y), size=n, replace=False)
    
    meta = {
        "regime": fetch_california_housing,
        "feature_names": california_housing.feature_names
    }
    return X[indices], y[indices], meta
