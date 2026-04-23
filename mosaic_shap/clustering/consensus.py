import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, List
import warnings
from collections import Counter
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
import shap

from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import math

warnings.filterwarnings('ignore')


# Chargement des données
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Nettoyage des outliers (identique)
X = X.loc[X['Population'] < 10000]
X = X.loc[X['AveOccup'] < 6]
X = X.loc[X['AveBedrms'] < 1.5]
X = X.loc[X['HouseAge'] < 50]
X = X.loc[(X['Latitude'] < 38.07) & (X['Latitude'] > 37.2)]
X = X.loc[(X['Longitude'] > -122.5) & (X['Longitude'] < -121.75)]
y = y[X.index]  # aligner y

# Entraînement d'un modèle
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Calcul des SHAP values
explainer = shap.TreeExplainer(model)
shap_values_array = explainer.shap_values(X)
shap_values = pd.DataFrame(shap_values_array, columns=X.columns, index=X.index)
print(f"Taille du dataset: {len(X)} échantillons, {X.shape[1]} features")
print(f"SHAP values shape: {shap_values.shape}")


class ShapBasedKmeansWrapper:
    """Wrapper Python pour ShapBasedKmeans qui évite les problèmes Cython"""
    
    def __init__(self, X: pd.DataFrame, progress_updated: Callable = None):
        self.X = X
        self.progress_updated = progress_updated or (lambda p, t: None)
    
    @staticmethod
    def scale(X, shap_values):
        """Scale les features par leurs valeurs SHAP"""
        std = X.std()
        std[std == 0] = 1
        return (X - X.mean()) / std * np.abs(shap_values).mean()
    
    def compute(self, X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters='auto') -> np.ndarray:
        """Calcule le clustering K-means basé sur SHAP"""
        # Scale
        x_scaled = self.scale(X, shap_values)
        
        # Concat
        X2 = pd.concat([x_scaled, shap_values], axis=1)
        
        # Clustering
        if n_clusters == 'auto':
            best_score = -np.inf
            best_clusters = None
            
            for k in range(2, min(30, len(X) // 2)):
                self.progress_updated(int(k / 30 * 100), 0)
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                clusters = km.fit_predict(X2)
                score = calinski_harabasz_score(X2, clusters)
                
                if score > best_score:
                    best_score = score
                    best_clusters = clusters
            
            return best_clusters
        else:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            return km.fit_predict(X2)


class ShapBasedHdbscanWrapper:
    """Wrapper Python pour ShapBasedHdbscan qui évite les problèmes Cython"""
    
    def __init__(self, X: pd.DataFrame, progress_updated: Callable = None):
        self.X = X
        self.progress_updated = progress_updated or (lambda p, t: None)
    
    @staticmethod
    def scale(X, shap_values):
        """Scale les features par leurs valeurs SHAP"""
        std = X.std()
        std[std == 0] = 1
        return (X - X.mean()) / std * np.abs(shap_values).mean()
    
    def compute(self, X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters='auto') -> np.ndarray:
        """Calcule le clustering HDBSCAN basé sur SHAP"""
        # Scale
        x_scaled = self.scale(X, shap_values)
        
        # Concat
        X2 = pd.concat([x_scaled, shap_values], axis=1)
        
        # Clustering
        if n_clusters == 'auto':
            hdbscan = HDBSCAN(min_cluster_size=int(math.sqrt(len(X))))
            return hdbscan.fit_predict(X2)
        else:
            aggc = AgglomerativeClustering(n_clusters=n_clusters)
            return aggc.fit_predict(X2)


class ShapBasedTomatoWrapper:
    """Wrapper Python pour ShapBasedTomato qui évite les problèmes Cython"""
    
    def __init__(self, X: pd.DataFrame, progress_updated: Callable = None):
        self.X = X
        self.progress_updated = progress_updated or (lambda p, t: None)
    
    @staticmethod
    def scale(X, shap_values):
        """Scale les features par leurs valeurs SHAP"""
        std = X.std()
        std[std == 0] = 1
        coef = np.abs(shap_values).mean() / std
        return (X - X.mean()) * coef
    
    def compute(self, X: pd.DataFrame, shap_values: pd.DataFrame, n_clusters='auto') -> np.ndarray:
        """Calcule le clustering TOMATO basé sur SHAP"""
        # Scale
        x_scaled = self.scale(X, shap_values)
        
        # Concat
        X2 = pd.concat([x_scaled, shap_values], axis=1)
        
        try:
            from tomaster import tomato
            # Clustering TOMATO
            if n_clusters == 'auto':
                return tomato(points=X2.values, k=20)
            else:
                return tomato(points=X2.values, n_clusters=n_clusters, k=20)
        except ImportError:
            print("   Module tomaster non disponible, utilisation de AgglomerativeClustering")
            # Fallback sur AgglomerativeClustering
            n_clust = n_clusters if n_clusters != 'auto' else max(2, int(math.sqrt(len(X))))
            aggc = AgglomerativeClustering(n_clusters=n_clust)
            return aggc.fit_predict(X2)


# ===================================================================
# CLASSE DE CONSENSUS
# ===================================================================

class TripleClusteringConsensus:
    """
    Classe pour réaliser un clustering consensus basé sur 3 algorithmes:
    - K-means (SHAP-based)
    - HDBSCAN (SHAP-based)  
    - TOMATO (SHAP-based)
    """
    
    def __init__(self, X: pd.DataFrame, progress_updated: Callable = None):
        self.X = X
        self.progress_updated = progress_updated or (lambda progress, elapsed_time: None)
        
        # Initialisation avec les wrappers Python
        self.kmeans = ShapBasedKmeansWrapper(X, self.progress_updated)
        self.hdbscan = ShapBasedHdbscanWrapper(X, self.progress_updated)
        self.tomato = ShapBasedTomatoWrapper(X, self.progress_updated)
        
    def compute_individual_clusterings(
        self, 
        X: pd.DataFrame, 
        shap_values: pd.DataFrame,
        n_clusters: int | str = 'auto'
    ) -> dict:
        """Calcule les clusterings individuels pour chaque algorithme"""
        
        print("Calcul du clustering K-means...")
        clusters_kmeans = self.kmeans.compute(X, shap_values, n_clusters)
        
        print("Calcul du clustering HDBSCAN...")
        clusters_hdbscan = self.hdbscan.compute(X, shap_values, n_clusters)
        
        print("Calcul du clustering TOMATO...")
        clusters_tomato = self.tomato.compute(X, shap_values, n_clusters)
        
        # Conversion en Series
        return {
            'kmeans': pd.Series(clusters_kmeans, index=X.index),
            'hdbscan': pd.Series(clusters_hdbscan, index=X.index),
            'tomato': pd.Series(clusters_tomato, index=X.index)
        }
    
    @staticmethod
    def consensus_voting(clusterings: dict) -> pd.Series:
        """
        Consensus par vote majoritaire.
        Pour chaque paire de points, on vote s'ils sont dans le même cluster.
        """
        n_samples = len(list(clusterings.values())[0])
        n_algos = len(clusterings)
        
        # Matrice de co-association
        co_matrix = np.zeros((n_samples, n_samples))
        
        for name, clusters in clusterings.items():
            clusters_array = clusters.values
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if clusters_array[i] == clusters_array[j] and clusters_array[i] != -1:
                        co_matrix[i, j] += 1
                        co_matrix[j, i] += 1
        
        # Normalisation
        co_matrix = co_matrix / n_algos
        
        # Clustering hiérarchique sur la matrice de consensus
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Conversion en distance
        distance_matrix = 1 - co_matrix
        
        # Linkage
        condensed_dist = distance_matrix[np.triu_indices(n_samples, k=1)]
        Z = linkage(condensed_dist, method='average')
        
        # Nombre de clusters basé sur la médiane des nombres de clusters individuels
        n_clusters_list = [len(c[c != -1].unique()) for c in clusterings.values()]
        n_clusters_consensus = int(np.median(n_clusters_list))
        n_clusters_consensus = max(2, n_clusters_consensus)  # Au moins 2 clusters
        
        # Extraction des clusters
        consensus_labels = fcluster(Z, n_clusters_consensus, criterion='maxclust') - 1
        
        return pd.Series(consensus_labels, index=list(clusterings.values())[0].index)
    
    def compute_consensus(
        self,
        X: pd.DataFrame,
        shap_values: pd.DataFrame,
        n_clusters: int | str = 'auto',
        method: str = 'voting'
    ) -> tuple:
        """
        Calcule le clustering consensus
        
        Args:
            X: DataFrame des features
            shap_values: DataFrame des valeurs SHAP
            n_clusters: nombre de clusters ou 'auto'
            method: 'voting'
        
        Returns:
            Tuple (consensus_labels, individual_clusterings)
        """
        # Calcul des clusterings individuels
        individual_clusterings = self.compute_individual_clusterings(
            X, shap_values, n_clusters
        )
        
        # Affichage des statistiques individuelles
        print("\nStatistiques des clusterings individuels:")
        for name, clusters in individual_clusterings.items():
            n_clusters_found = len(clusters.unique())
            n_noise = (clusters == -1).sum()
            print(f"  {name:10s}: {n_clusters_found} clusters ({n_noise} points non assignés)")
        
        # Calcul du consensus
        print(f"\nCalcul du consensus (méthode: {method})...")
        consensus = self.consensus_voting(individual_clusterings)
        
        print(f"Consensus: {len(consensus.unique())} clusters")
        
        return consensus, individual_clusterings

print("\n" + "="*60)
print("TRIPLE CLUSTERING CONSENSUS")
print("="*60)

consensus_algo = TripleClusteringConsensus(X)

# Calcul du consensus
consensus_labels, individual_clusterings = consensus_algo.compute_consensus(
    X, 
    shap_values, 
    n_clusters='auto',
    method='voting'
)


# ===================================================================
# MÉTRIQUES
# ===================================================================

def compute_clustering_metrics(X: pd.DataFrame, labels: pd.Series, name: str):
    """Calcule diverses métriques pour évaluer la qualité du clustering"""
    
    # Filtrer les points non assignés (-1)
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    if len(labels_valid.unique()) < 2:
        print(f"  {name}: Pas assez de clusters pour calculer les métriques")
        return None
    
    metrics = {
        'n_clusters': len(labels.unique()),
        'n_noise': (labels == -1).sum(),
        'silhouette': silhouette_score(X_valid, labels_valid) if len(labels_valid.unique()) > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X_valid, labels_valid) if len(labels_valid.unique()) > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X_valid, labels_valid) if len(labels_valid.unique()) > 1 else np.nan,
    }
    
    return metrics


# Calcul des métriques pour le consensus
print("\n" + "="*60)
print("MÉTRIQUES DE QUALITÉ DES CLUSTERINGS")
print("="*60)

consensus_metrics = compute_clustering_metrics(X, consensus_labels, "Consensus")

print("\nCONSENSUS:")
if consensus_metrics:
    for metric, value in consensus_metrics.items():
        print(f"  {metric:20s}: {value:.4f}" if isinstance(value, float) else f"  {metric:20s}: {value}")

# Métriques pour les clusterings individuels
print("\nCLUSTERINGS INDIVIDUELS:")
individual_metrics = {}
for name, clusters in individual_clusterings.items():
    metrics = compute_clustering_metrics(X, clusters, name)
    individual_metrics[name] = metrics
    print(f"\n  {name.upper()}:")
    if metrics:
        for metric, value in metrics.items():
            print(f"    {metric:20s}: {value:.4f}" if isinstance(value, float) else f"    {metric:20s}: {value}")


# Accord entre les méthodes
print("\n" + "="*60)
print("ACCORD ENTRE LES MÉTHODES (ARI & NMI)")
print("="*60)

# Calcul de l'ARI et NMI entre toutes les paires
methods = list(individual_clusterings.keys()) + ['consensus']
all_clusterings = {**individual_clusterings, 'consensus': consensus_labels}

agreement_matrix_ari = pd.DataFrame(
    np.zeros((len(methods), len(methods))),
    index=methods,
    columns=methods  
)

agreement_matrix_nmi = pd.DataFrame(
    np.zeros((len(methods), len(methods))),
    index=methods,
    columns=methods
)

for i, method1 in enumerate(methods):
    for j, method2 in enumerate(methods):
        if i <= j:
            labels1 = all_clusterings[method1]
            labels2 = all_clusterings[method2]
            
            # Filtrer les points communs non-noise
            valid_mask = (labels1 != -1) & (labels2 != -1)
            
            if valid_mask.sum() > 0:
                ari = adjusted_rand_score(labels1[valid_mask], labels2[valid_mask])
                nmi = normalized_mutual_info_score(labels1[valid_mask], labels2[valid_mask])
                
                agreement_matrix_ari.loc[method1, method2] = ari
                agreement_matrix_ari.loc[method2, method1] = ari
                
                agreement_matrix_nmi.loc[method1, method2] = nmi
                agreement_matrix_nmi.loc[method2, method1] = nmi

print("\nAdjusted Rand Index (ARI):")
print(agreement_matrix_ari.round(3))

print("\nNormalized Mutual Information (NMI):")
print(agreement_matrix_nmi.round(3))


# ===================================================================
# VISUALISATIONS
# ===================================================================

print("\n" + "="*60)
print("GÉNÉRATION DES VISUALISATIONS")
print("="*60)

# Visualisation des clusterings
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# Sélection de 2 features pour la visualisation
feature_x = 'HouseAge'
feature_y = 'AveRooms'

# Plot 1: K-means
scatter1 = axes[0].scatter(X[feature_x], X[feature_y], 
                          c=individual_clusterings['kmeans'], 
                          cmap='tab10', alpha=0.6, s=30)
axes[0].set_title('K-means (SHAP-based)', fontsize=12, fontweight='bold')
axes[0].set_xlabel(feature_x)
axes[0].set_ylabel(feature_y)
plt.colorbar(scatter1, ax=axes[0])

# Plot 2: HDBSCAN
scatter2 = axes[1].scatter(X[feature_x], X[feature_y], 
                          c=individual_clusterings['hdbscan'], 
                          cmap='tab10', alpha=0.6, s=30)
axes[1].set_title('HDBSCAN (SHAP-based)', fontsize=12, fontweight='bold')
axes[1].set_xlabel(feature_x)
axes[1].set_ylabel(feature_y)
plt.colorbar(scatter2, ax=axes[1])

# Plot 3: TOMATO
scatter3 = axes[2].scatter(X[feature_x], X[feature_y], 
                          c=individual_clusterings['tomato'], 
                          cmap='tab10', alpha=0.6, s=30)
axes[2].set_title('TOMATO (SHAP-based)', fontsize=12, fontweight='bold')
axes[2].set_xlabel(feature_x)
axes[2].set_ylabel(feature_y)
plt.colorbar(scatter3, ax=axes[2])

# Plot 4: Consensus
scatter4 = axes[3].scatter(X[feature_x], X[feature_y], 
                          c=consensus_labels, 
                          cmap='tab10', alpha=0.6, s=30)
axes[3].set_title('CONSENSUS (Triple Clustering)', fontsize=12, fontweight='bold')
axes[3].set_xlabel(feature_x)
axes[3].set_ylabel(feature_y)
plt.colorbar(scatter4, ax=axes[3])

plt.tight_layout()
plt.savefig('triple_clustering_comparison.png', dpi=300, bbox_inches='tight')
print(" Sauvegardé: triple_clustering_comparison.png")
plt.show()

"""
# Heatmap des accords
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(agreement_matrix_ari, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'ARI Score'})
ax1.set_title('Adjusted Rand Index entre les méthodes', fontsize=14, fontweight='bold')

sns.heatmap(agreement_matrix_nmi, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'NMI Score'})
ax2.set_title('Normalized Mutual Information entre les méthodes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('agreement_heatmaps.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: agreement_heatmaps.png")
plt.show()


# Comparaison des métriques
all_metrics = {
    'K-means': individual_metrics['kmeans'],
    'HDBSCAN': individual_metrics['hdbscan'],
    'TOMATO': individual_metrics['tomato'],
    'Consensus': consensus_metrics
}

metrics_df = pd.DataFrame(all_metrics).T

print("\n" + "="*60)
print("TABLEAU RÉCAPITULATIF DES MÉTRIQUES")
print("="*60)
print(metrics_df.round(4))

# Visualisation des métriques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
titles = ['Silhouette Score\n(plus élevé = meilleur)', 
          'Calinski-Harabasz Score\n(plus élevé = meilleur)',
          'Davies-Bouldin Score\n(plus faible = meilleur)']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    values = metrics_df[metric].values
    methods = metrics_df.index.tolist()
    
    bars = axes[idx].bar(methods, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Score')
    axes[idx].tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print(" Sauvegardé: metrics_comparison.png")
plt.show()


# Conclusions
print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)

print("\n Résumé des résultats:")
print(f"  • Nombre de clusters (K-means): {individual_metrics['kmeans']['n_clusters']}")
print(f"  • Nombre de clusters (HDBSCAN): {individual_metrics['hdbscan']['n_clusters']}")
print(f"  • Nombre de clusters (TOMATO): {individual_metrics['tomato']['n_clusters']}")
print(f"  • Nombre de clusters (Consensus): {consensus_metrics['n_clusters']}")

print("\n Meilleure méthode selon les métriques:")

best_silhouette = metrics_df['silhouette'].idxmax()
print(f"  • Silhouette Score: {best_silhouette} ({metrics_df.loc[best_silhouette, 'silhouette']:.4f})")

best_ch = metrics_df['calinski_harabasz'].idxmax()
print(f"  • Calinski-Harabasz: {best_ch} ({metrics_df.loc[best_ch, 'calinski_harabasz']:.4f})")

best_db = metrics_df['davies_bouldin'].idxmin()
print(f"  • Davies-Bouldin: {best_db} ({metrics_df.loc[best_db, 'davies_bouldin']:.4f})")

print("\nAvantages du consensus:")
print("  • Robustesse accrue par combinaison de 3 algorithmes différents")
print("  • Réduction du biais algorithmique")
print("  • Capture de structures de clustering complémentaires")

consensus_agreement = agreement_matrix_ari.loc['consensus', ['kmeans', 'hdbscan', 'tomato']].mean()
print(f"\n  • Accord moyen du consensus avec les méthodes individuelles (ARI): {consensus_agreement:.4f}")

print("\n" + "="*60)
print("ANALYSE TERMINÉE ")
print("="*60)
"""