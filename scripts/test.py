#!/usr/bin/env python3
"""
California Housing - Interactions SHAP d'ordre 2 avec clustering avancé.
Améliorations :
- Échantillonnage stratifié
- PCA optionnelle avant UMAP
- Évaluation des clusters (silhouette, Davies‑Bouldin, stabilité)
- Optimisation automatique de HDBSCAN (recherche par grille)
- Visualisations enrichies : heatmap des interactions discriminantes, barplots par cluster
- Cache des interactions SHAP (fichier .npy)
- Fond de carte contextily (si disponible)
- Sauvegarde des métriques dans un fichier JSON
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
import shap
import umap
import hdbscan
from scipy.stats import f_oneway
from joblib import Parallel, delayed
import time

# Options silencieuses
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration (peut être remplacée par un fichier YAML)
# ----------------------------------------------------------------------
CONFIG = {
    "seed": 0,
    "n_samples": 3000,          # None pour tout utiliser
    "test_size": 0.2,           # pour échantillonnage stratifié (sous-ensemble)
    "use_pca": True,            # PCA avant UMAP
    "pca_dim": 30,              # dimension PCA (si None, garde 95% de variance)
    "umap_n_neighbors": 15,     # pour UMAP (peut être exploré)
    "umap_min_dist": 0.1,
    "hdbscan_min_cluster_size": 20,   # utilisé si pas de grid search
    "hdbscan_min_samples": None,
    "grid_search_hdbscan": True,      # active la recherche des meilleurs paramètres
    "grid_hdbscan_cluster_size": [10, 20, 40, 80],
    "grid_hdbscan_min_samples": [5, 10, 20],
    "n_stability_runs": 10,      # nombre de sous-échantillonnages pour stabilité
    "stability_sample_frac": 0.8,
    "cache_dir": "cache",
    "output_dir": "figures/ordre2_cluster_test_advanced",
    "top_interactions_heatmap": 30,   # nombre d'interactions affichées dans la heatmap
    "top_interactions_per_cluster": 10,
}

# Création des répertoires
os.makedirs(CONFIG["cache_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    # umap et hdbscan utilisent aussi le seed via paramètre

def stratified_sample(X, y, n_samples, test_size, seed):
    """Échantillonnage stratifié sur y (prix discrétisé) pour garder distribution."""
    if n_samples is None or n_samples >= len(X):
        return X, y
    # Discrétiser y en 5 classes pour stratification
    y_binned = pd.cut(y, bins=5, labels=False)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - n_samples/len(X), random_state=seed)
    idx_train, _ = next(sss.split(X, y_binned))
    return X[idx_train], y[idx_train]

def compute_shap_interactions(model, X, cache_key):
    """Calcule les interactions SHAP d'ordre 2 avec cache."""
    cache_path = os.path.join(CONFIG["cache_dir"], f"{cache_key}.npy")
    if os.path.exists(cache_path):
        print(f"Chargement des interactions depuis {cache_path}")
        return np.load(cache_path)
    print("Calcul des interactions SHAP d'ordre 2...")
    start = time.time()
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X)
    if isinstance(shap_interaction, list):
        shap_interaction = shap_interaction[0]
    np.save(cache_path, shap_interaction)
    print(f"Calcul terminé en {time.time()-start:.2f}s, sauvegardé dans {cache_path}")
    return shap_interaction

def vectorize_interactions(shap_interaction):
    """Retourne matrice (n_samples, n_pairs) et noms des paires."""
    p = shap_interaction.shape[1]
    iu = np.triu_indices(p, k=1)
    n_pairs = len(iu[0])
    vectors = shap_interaction[:, iu[0], iu[1]]
    return vectors, iu

def evaluate_clustering(vectors, labels, umap_coords=None):
    """Calcule silhouette (sur vecteurs originaux et/ou UMAP) et Davies-Bouldin."""
    # Filtrer les points non bruités (label >= 0)
    mask = labels >= 0
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": np.unique(labels).size}
    # Sur vecteurs originaux
    try:
        sil = silhouette_score(vectors[mask], labels[mask])
        db = davies_bouldin_score(vectors[mask], labels[mask])
    except:
        sil = db = None
    res = {"silhouette": sil, "davies_bouldin": db, "n_clusters": len(np.unique(labels[mask]))}
    # Optionnel : sur UMAP
    if umap_coords is not None:
        try:
            sil_umap = silhouette_score(umap_coords[mask], labels[mask])
            res["silhouette_umap"] = sil_umap
        except:
            pass
    return res

def stability_analysis(vectors, method, n_runs, sample_frac, seed):
    """Évalue la stabilité du clustering par sous-échantillonnage (ARI)."""
    n_samples = vectors.shape[0]
    labels_list = []
    ari_list = []
    rng = np.random.RandomState(seed)
    
    for run in range(n_runs):
        idx = rng.choice(n_samples, size=int(n_samples*sample_frac), replace=False)
        vec_sub = vectors[idx]
        
        # PCA optionnelle (même logique que dans le main)
        if CONFIG["use_pca"]:
            # Calculer le nombre de composantes maximal possible
            max_comp = min(vec_sub.shape[1], CONFIG["pca_dim"]) if CONFIG["pca_dim"] is not None else None
            pca = PCA(n_components=max_comp, random_state=seed)
            vec_sub = pca.fit_transform(vec_sub)
        
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=CONFIG["umap_n_neighbors"],
                            min_dist=CONFIG["umap_min_dist"])
        emb = reducer.fit_transform(vec_sub)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
                                    min_samples=CONFIG["hdbscan_min_samples"])
        labels_sub = clusterer.fit_predict(emb)
        labels_list.append(labels_sub)
        
        if run > 0:
            # Comparer avec le premier run (indice commun)
            common = np.intersect1d(idx, idx_prev)
            if len(common) > 1:
                # Re-indexer
                map1 = {i: labels_list[0][pos] for pos, i in enumerate(idx_prev) if i in common}
                map2 = {i: labels_sub[pos] for pos, i in enumerate(idx) if i in common}
                labels1 = [map1[i] for i in common]
                labels2 = [map2[i] for i in common]
                ari = adjusted_rand_score(labels1, labels2)
                ari_list.append(ari)
        idx_prev = idx
    
    stability = np.mean(ari_list) if ari_list else None
    return stability

def grid_search_hdbscan(vectors, param_grid):
    """Recherche les meilleurs paramètres HDBSCAN basée sur la silhouette moyenne."""
    best_score = -1
    best_params = None
    # On peut aussi faire une validation croisée simple
    for min_size in param_grid["min_cluster_size"]:
        for min_samples in param_grid["min_samples"]:
            # Appliquer UMAP avec les paramètres par défaut (ou fixes)
            # Ici, on suppose que les vecteurs sont déjà réduits (UMAP effectué une fois)
            # Pour rester simple, on utilise les vecteurs après PCA+UMAP (déjà faits)
            # Mais dans la recherche, on devrait peut-être refaire UMAP ? Non, on fixe UMAP.
            # Donc on utilise les coordonnées UMAP déjà calculées (emb).
            # Cette fonction sera appelée après réduction, on passe donc les vecteurs UMAP.
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_samples)
            labels = clusterer.fit_predict(vectors)
            if len(np.unique(labels[labels>=0])) < 2:
                continue
            try:
                sil = silhouette_score(vectors[labels>=0], labels[labels>=0])
                if sil > best_score:
                    best_score = sil
                    best_params = {"min_cluster_size": min_size, "min_samples": min_samples}
            except:
                continue
    return best_params, best_score

def plot_umap(emb, color, title, filename, colorbar_label=None, discrete=False):
    plt.figure(figsize=(7,6))
    if discrete:
        # color discret (cluster)
        unique = np.unique(color)
        for c in unique:
            mask = color == c
            plt.scatter(emb[mask,0], emb[mask,1], s=8, alpha=0.7, label=f"Cluster {c}" if c>=0 else "Bruit")
        plt.legend(markerscale=3, fontsize=8)
    else:
        sc = plt.scatter(emb[:,0], emb[:,1], c=color, s=8, cmap="viridis", alpha=0.7)
        if colorbar_label:
            plt.colorbar(sc, label=colorbar_label)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], filename), dpi=180)
    plt.close()

def plot_map(df, color_col, title, filename, discrete=False):
    fig, ax = plt.subplots(figsize=(8,8))
    lon, lat = df["Longitude"].values, df["Latitude"].values
    if discrete:
        unique = np.unique(df[color_col].values)
        for c in unique:
            mask = df[color_col] == c
            ax.scatter(lon[mask], lat[mask], s=5, alpha=0.6, label=f"Cluster {c}" if c>=0 else "Bruit")
        ax.legend(markerscale=2, fontsize=8)
    else:
        sc = ax.scatter(lon, lat, c=df[color_col], s=5, cmap="viridis", alpha=0.6)
        plt.colorbar(sc, label=color_col)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lon.min()-0.5, lon.max()+0.5)
    ax.set_ylim(lat.min()-0.5, lat.max()+0.5)
    # Ajouter fond de carte si contextily disponible
    try:
        import contextily as cx
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_merc, y_merc = t.transform(lon, lat)
        ax.set_xlim(x_merc.min()-1e4, x_merc.max()+1e4)
        ax.set_ylim(y_merc.min()-1e4, y_merc.max()+1e4)
        cx.add_basemap(ax, crs="EPSG:3857", zoom=6, alpha=0.8)
    except:
        pass
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], filename), dpi=180)
    plt.close()

def plot_heatmap_discriminant(df, interaction_names, cluster_col, top_k):
    """Heatmap des interactions les plus discriminantes entre clusters (ANOVA)."""
    labels = df[cluster_col].values
    clusters = sorted([c for c in np.unique(labels) if c >= 0])
    if len(clusters) < 2:
        return
    shap_cols = [c for c in df.columns if c.startswith("shap2_")]
    # Calculer ANOVA univariée pour chaque interaction
    f_vals = []
    for col in shap_cols:
        groups = [df.loc[labels==c, col].values for c in clusters]
        # Vérifier que chaque groupe a au moins 2 points
        if all(len(g)>=2 for g in groups):
            f, p = f_oneway(*groups)
            f_vals.append(f)
        else:
            f_vals.append(0)
    # Prendre les top_k selon F
    top_idx = np.argsort(f_vals)[-top_k:][::-1]
    selected_cols = [shap_cols[i] for i in top_idx]
    selected_names = [interaction_names[i] for i in top_idx]
    # Matrice des moyennes
    M = np.zeros((len(clusters), len(selected_cols)))
    for i, c in enumerate(clusters):
        mask = labels == c
        M[i] = df.loc[mask, selected_cols].mean(axis=0).values
    fig, ax = plt.subplots(figsize=(max(8, 0.4*len(selected_names)), max(4, 0.4*len(clusters))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    plt.colorbar(im, ax=ax, label="Moyenne SHAP interaction")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters])
    ax.set_xticks(range(len(selected_names)))
    ax.set_xticklabels(selected_names, rotation=90, ha="right", fontsize=8)
    ax.set_title(f"Clusters × interactions (top {top_k} discriminantes)")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "heatmap_discriminant.png"), dpi=180)
    plt.close()

def plot_cluster_barplots(df, interaction_names, cluster_col, top_k):
    """Pour chaque cluster, barplot des interactions les plus fortes (en valeur absolue moyenne)."""
    labels = df[cluster_col].values
    clusters = sorted([c for c in np.unique(labels) if c >= 0])
    shap_cols = [c for c in df.columns if c.startswith("shap2_")]
    for c in clusters:
        mask = labels == c
        mean_abs = np.abs(df.loc[mask, shap_cols].mean(axis=0).values)
        top_idx = np.argsort(mean_abs)[-top_k:][::-1]
        top_names = [interaction_names[i] for i in top_idx]
        top_vals = mean_abs[top_idx]
        plt.figure(figsize=(8,5))
        plt.barh(range(len(top_vals)), top_vals)
        plt.yticks(range(len(top_vals)), top_names)
        plt.xlabel("Moyenne |SHAP interaction|")
        plt.title(f"Cluster {c} – top {top_k} interactions (valeur absolue)")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], f"cluster_{c}_top_interactions.png"), dpi=180)
        plt.close()

# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------
def main():
    set_seed(CONFIG["seed"])
    
    # 1. Charger données
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    feature_names = data.feature_names
    
    # 2. Échantillonnage stratifié
    X, y = stratified_sample(X_full, y_full, CONFIG["n_samples"], CONFIG["test_size"], CONFIG["seed"])
    print(f"Données: {X.shape[0]} échantillons, {X.shape[1]} features")
    
    # 3. Entraînement du modèle
    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, min_samples_split=5,
        learning_rate=0.05, random_state=CONFIG["seed"]
    )
    model.fit(X, y)
    
    # 4. Calcul des interactions SHAP (avec cache)
    cache_key = f"shap_interactions_{X.shape[0]}_{hash(X.tobytes())}"  # simplifié
    shap_interaction = compute_shap_interactions(model, X, cache_key)
    
    # 5. Vectorisation
    interaction_vectors, iu = vectorize_interactions(shap_interaction)
    interaction_names = [f"{feature_names[i]} × {feature_names[j]}" for i,j in zip(iu[0], iu[1])]
    
    # 6. Réduction PCA (optionnelle)
    if CONFIG["use_pca"]:
        n_pairs = interaction_vectors.shape[1]
        n_comp = min(n_pairs, CONFIG["pca_dim"]) if CONFIG["pca_dim"] is not None else None
        pca = PCA(n_components=n_comp, random_state=CONFIG["seed"])
        vectors_reduced = pca.fit_transform(interaction_vectors)
        print(f"PCA: {n_comp} composantes, variance expliquée = {pca.explained_variance_ratio_.sum():.2%}")
    
    # 7. UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=CONFIG["seed"],
                             n_neighbors=CONFIG["umap_n_neighbors"],
                             min_dist=CONFIG["umap_min_dist"])
    emb = umap_reducer.fit_transform(vectors_reduced)
    
    # 8. Clustering avec HDBSCAN (option grid search)
    if CONFIG["grid_search_hdbscan"]:
        # Préparer grille
        param_grid = {
            "min_cluster_size": CONFIG["grid_hdbscan_cluster_size"],
            "min_samples": CONFIG["grid_hdbscan_min_samples"]
        }
        best_params, best_score = grid_search_hdbscan(emb, param_grid)
        if best_params is not None:
            CONFIG["hdbscan_min_cluster_size"] = best_params["min_cluster_size"]
            CONFIG["hdbscan_min_samples"] = best_params["min_samples"]
            print(f"Grid search HDBSCAN -> min_cluster_size={best_params['min_cluster_size']}, min_samples={best_params['min_samples']}, silhouette={best_score:.3f}")
        else:
            print("Grid search n'a pas trouvé de paramètre valide, utilisation des valeurs par défaut.")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
                                min_samples=CONFIG["hdbscan_min_samples"])
    labels = clusterer.fit_predict(emb)
    print(f"Clusters: {len(np.unique(labels[labels>=0]))} clusters, bruit: {(labels==-1).sum()}")
    
    # 9. Évaluation des clusters
    metrics = evaluate_clustering(interaction_vectors, labels, emb)
    print("Métriques de clustering:")
    for k,v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.3f}" if isinstance(v,float) else f"  {k}: {v}")
    
    # 10. Stabilité par sous-échantillonnage
    # On réutilise le vecteur après PCA (si utilisé) et UMAP? Pour la stabilité, il faudrait refaire le pipeline complet,
    # mais on peut se contenter des vecteurs d'interactions bruts pour accélérer. On utilise les paramètres finaux.
    print("Analyse de stabilité...")
    stability = stability_analysis(interaction_vectors, "hdbscan", CONFIG["n_stability_runs"],
                                   CONFIG["stability_sample_frac"], CONFIG["seed"])
    if stability is not None:
        print(f"Stabilité (ARI moyenne): {stability:.3f}")
        metrics["stability_ari"] = stability
    
    # Sauvegarde des métriques dans un JSON
    with open(os.path.join(CONFIG["output_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 11. Construction du DataFrame pour visualisations et CSV
    df = pd.DataFrame(X, columns=[f"feat_{name}" for name in feature_names])
    df["price"] = y
    df["Latitude"] = X[:, 6]
    df["Longitude"] = X[:, 7]
    for j in range(interaction_vectors.shape[1]):
        df[f"shap2_{j}"] = interaction_vectors[:, j]
    df["umap_1"] = emb[:, 0]
    df["umap_2"] = emb[:, 1]
    df["cluster"] = labels
    df.to_csv(os.path.join(CONFIG["output_dir"], "shap_clusters.csv"), index=False)
    
    # 12. Visualisations
    plot_umap(emb, df["price"], "UMAP des interactions SHAP – couleur = prix", "umap_price.png", colorbar_label="Prix (×100k $)")
    plot_umap(emb, df["cluster"], "UMAP des interactions SHAP – couleur = cluster", "umap_cluster.png", discrete=True)
    plot_map(df, "price", "Carte Californie – couleur = prix", "map_price.png", discrete=False)
    plot_map(df, "cluster", "Carte Californie – couleur = cluster", "map_cluster.png", discrete=True)
    plot_heatmap_discriminant(df, interaction_names, "cluster", CONFIG["top_interactions_heatmap"])
    plot_cluster_barplots(df, interaction_names, "cluster", CONFIG["top_interactions_per_cluster"])
    
    # 13. Sauvegarde des paramètres de configuration (pour reproductibilité)
    with open(os.path.join(CONFIG["output_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\nTous les résultats sont dans {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()