import numpy as np

def make_dataset_overlap_scores_but_separable_interactions(n=1000, p_noise=4, seed=0, sigma=0.9):
    """Scores A/B overlap in score space, but regimes are separable in explainability space."""
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 2, size=n)

    X1 = rng.uniform(0, 1, size=n)*10-5 + 0.2 * rng.normal(size=n)
    X2 = rng.uniform(0, 1, size=n)*10-5 + 0.2 * rng.normal(size=n)
    X3 = rng.uniform(0, 1, size=n)*10-5 + 0.2 * rng.normal(size=n)
    X4 = rng.uniform(0, 1, size=n)*10-5 + 0.2 * rng.normal(size=n)
    X5 = rng.uniform(0, 1, size=n)*10-5 + 0.2 * rng.normal(size=n)
    X6 = rng.uniform(0, 1, size=n)*10-5 + 0.2 * rng.normal(size=n)

    U = rng.uniform(size=(8, n))
    interaction = X1 * X2

    AdditifA = -6*X1 + 6*X2 + U[0]*X3 + U[2]*X4 + U[4]*X5 + U[6]*X6
    AdditifB =  6*X1 - 6*X2 + U[1]*X3 + U[3]*X4 + U[5]*X5 + U[7]*X6
    eps = sigma * rng.normal(size=n)

    score_A =  6.0 * (-interaction) + AdditifA + eps
    score_B =  6.0 * ( interaction) + AdditifB + eps
    score = np.where(regime == 0, score_A, score_B)

    prob = np.exp(score_B) / (np.exp(score_A) + np.exp(score_B))
    y = rng.binomial(1, prob)

    noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X1, X2, X3, X4, X5, X6, noise])
    feature_names = [f"X{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, p_noise+1)]

    meta = {
        "regime": regime,
        "feature_names": feature_names,
        "score_A": score_A,
        "score_B": score_B,
        "score": score
    }
    return X, y, meta
