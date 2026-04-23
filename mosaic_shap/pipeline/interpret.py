import numpy as np

def top_interactions_per_cluster(shap2, labels, feature_names, top_k=5):
    """
    Pour chaque cluster, retourne les top_k paires (i,j) les plus discriminantes.
    Discriminant = |mean_cluster(φᵢⱼ)| - |mean_global(φᵢⱼ)|
    """
    results = {}
    M = shap2.shape[1]
    global_mean = np.abs(shap2).mean(axis=0)   # (M, M)
    
    for label in np.unique(labels):
        if label < 0:
            continue
        mask = labels == label
        cluster_mean = np.abs(shap2[mask]).mean(axis=0)   # (M, M)
        discriminance = cluster_mean - global_mean        # (M, M)
        
        # Top-k paires
        idx = np.dstack(np.unravel_index(
            np.argsort(discriminance.ravel())[::-1], (M, M)
        ))[0]
        top_pairs = [(feature_names[i], feature_names[j], discriminance[i,j])
                     for i, j in idx[:top_k] if i < j]
        results[label] = top_pairs
    
    return results