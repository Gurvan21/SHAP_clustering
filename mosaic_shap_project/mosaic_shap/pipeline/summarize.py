import numpy as np
import pandas as pd

def summarize_order1_by_cluster(phi1: np.ndarray, labels: np.ndarray, feature_names, topk: int = 8):
    rows=[]
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = labels == c
        mean_abs = np.mean(np.abs(phi1[idx]), axis=0)
        top = np.argsort(-mean_abs)[:topk]
        feats=[(feature_names[i], float(mean_abs[i])) for i in top]
        rows.append({"cluster": int(c), "size": int(idx.sum()), "top_features": feats})
    return pd.DataFrame(rows)

def summarize_interactions_by_cluster(Z: np.ndarray, labels: np.ndarray, iu, feature_names, topk: int = 8):
    """Z is vectorized upper-triangle interactions (n,d). Values can be signed.
    We rank by mean absolute value, but we also report mean signed value.
    """
    rows=[]
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = labels == c
        mean = np.mean(Z[idx], axis=0)
        mean_abs = np.mean(np.abs(Z[idx]), axis=0)
        top = np.argsort(-mean_abs)[:topk]
        pairs=[]
        for t in top:
            i, j = int(iu[0][t]), int(iu[1][t])
            pairs.append(((feature_names[i], feature_names[j]), float(mean[t]), float(mean_abs[t])))
        rows.append({"cluster": int(c), "size": int(idx.sum()), "top_interactions": pairs})
    return pd.DataFrame(rows)
