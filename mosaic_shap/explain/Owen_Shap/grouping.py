import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import pdist
from itertools import combinations, permutations
from math import factorial
from typing import List, Dict, Tuple, Optional

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import shap
import scipy

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

def discover_groups_from_correlation(
    X: np.ndarray,
    K: int,
    method: str = 'ward',
    feature_names: List[str] = None,
    plot: bool = True
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Découvre des groupes de features par clustering hiérarchique sur
    la matrice de corrélation de X.

    Parameters
    ----------
    X : np.ndarray, shape (N, M)
    K : int — nombre de groupes souhaité
    method : str — méthode de linkage pour scipy
    plot : bool — afficher le dendrogramme

    Returns
    -------
    groups : list of list of int
    labels : np.ndarray, shape (M,) — label de groupe pour chaque feature

    Notes
    -----
    À intégrer dans mosaic_shap/grouping.py
    """
    corr = np.corrcoef(X.T)  # (M, M)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)

    Z = linkage(pdist(X.T, metric = 'correlation'), method='ward')
    labels = fcluster(Z, t=K, criterion='maxclust')  # 1-indexed
    groups = [np.where(labels == k)[0].tolist() for k in range(1, K + 1)]
    groups = [g for g in groups if len(g) > 0]  # enlever groupes vides

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Dendrogramme
        dendrogram(
            Z,
            labels=feature_names or [str(i) for i in range(X.shape[1])],
            ax=axes[0], color_threshold=0,
            leaf_rotation=45
        )
        axes[0].set_title(f"Dendrogramme (linkage={method})")
        axes[0].axhline(
            y=Z[-(K-1), 2], color='red', linestyle='--',
            label=f'Cut → {K} groupes'
        )
        axes[0].legend(fontsize=9)

        # Matrice de corrélation réordonnée
        reorder = [f for g in groups for f in g]
        corr_reord = corr[np.ix_(reorder, reorder)]
        fn = feature_names or [str(i) for i in range(X.shape[1])]
        sns.heatmap(
            corr_reord,
            xticklabels=[fn[i] for i in reorder],
            yticklabels=[fn[i] for i in reorder],
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=axes[1], square=True, annot=True, fmt='.1f', annot_kws={'size': 7}
        )
        axes[1].set_title("Corrélation (features réordonnées par groupe)")

        # Cadres de groupe
        pos = 0
        for g in groups:
            for ax in [axes[1]]:
                rect = plt.Rectangle(
                    (pos, pos), len(g), len(g),
                    fill=False, edgecolor='black', lw=2.5
                )
                ax.add_patch(rect)
            pos += len(g)

        plt.tight_layout()
        plt.savefig("figures/grouping_correlation.png", bbox_inches='tight')
        plt.show()

    return groups, labels


def discover_groups_from_shap(
    shap_vals: np.ndarray,
    K: int,
    feature_names: List[str] = None,
    plot: bool = True
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Découvre des groupes de features par clustering hiérarchique sur
    la corrélation des SHAP values (espace des explications).

    Notes
    -----
    À intégrer dans mosaic_shap/grouping.py
    """
    corr = np.corrcoef(shap_vals.T)  # (M, M)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    #dist = (dist + dist.T) / 2
    
    Z = linkage(pdist(X.T, metric = 'correlation'), method='ward')
    labels = fcluster(Z, t=K, criterion='maxclust')
    groups = [np.where(labels == k)[0].tolist() for k in range(1, K + 1)]
    groups = [g for g in groups if len(g) > 0]

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        dendrogram(
            Z,
            labels=feature_names or [str(i) for i in range(shap_vals.shape[1])],
            ax=ax, color_threshold=0, leaf_rotation=45
        )
        ax.axhline(
            y=Z[-(K-1), 2], color='darkorange', linestyle='--',
            label=f'Cut → {K} groupes'
        )
        ax.legend(fontsize=9)
        ax.set_title("Dendrogramme sur les SHAP values (espace des explications)")
        plt.tight_layout()
        plt.savefig("figures/grouping_shap.png", bbox_inches='tight')
        plt.show()

    return groups, labels

def discover_two_level_hierarchy(
    shap_vals: np.ndarray,
    K_coarse: int,
    K_fine: int,
    feature_names: List[str] = None,
    plot: bool = True
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Découvre automatiquement une hiérarchie à 2 niveaux sur les features
    en clusterant leurs SHAP values (espace des explications).

    Étapes :
    1. Calcul de la distance inter-features dans l'espace des SHAP values
    2. Clustering hiérarchique (Ward)
    3. Coupe à K_coarse → groupes grossiers
    4. Coupe à K_fine   → groupes fins (raffinement des groupes grossiers)

    Parameters
    ----------
    shap_vals : np.ndarray, shape (N, M)
    K_coarse : int — nombre de groupes grossiers
    K_fine   : int — nombre de groupes fins (>= K_coarse)

    Returns
    -------
    coarse_groups : list of list of int
    fine_groups   : list of list of int

    Notes
    -----
    À intégrer dans mosaic_shap/grouping.py
    """
    assert K_fine >= K_coarse, "K_fine doit être >= K_coarse"
    
    data = fetch_california_housing(as_frame=True)
    X_full, y_full = data.data, data.target
    feature_names = list(X_full.columns)
    M = len(feature_names)  # 8 features


    N_SAMPLE = 500
    idx = np.random.choice(len(X_full), N_SAMPLE, replace=False)
    X = X_full.iloc[idx].reset_index(drop=True)
    y = y_full.iloc[idx].reset_index(drop=True)


    # Distance dans l'espace des explications
    corr = np.corrcoef(shap_vals.T)  # (M, M)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    Z = linkage(pdist(X.T, metric = 'correlation'), method='ward')

    # Coupures
    labels_coarse = fcluster(Z, t=K_coarse, criterion='maxclust')
    labels_fine   = fcluster(Z, t=K_fine,   criterion='maxclust')

    coarse_groups = [np.where(labels_coarse == k)[0].tolist() for k in range(1, K_coarse+1)]
    fine_groups   = [np.where(labels_fine   == k)[0].tolist() for k in range(1, K_fine+1)]
    # Enlever groupes vides
    coarse_groups = [g for g in coarse_groups if g]
    fine_groups   = [g for g in fine_groups   if g]

    # Vérification cohérence hiérarchique
    for fg in fine_groups:
        coarse_ids = set(labels_coarse[f]-1 for f in fg)
        if len(coarse_ids) > 1:
            # Un groupe fin chevauche deux groupes grossiers : on l'assigne au plus fréquent
            dominant_coarse = max(coarse_ids, key=lambda k: sum(labels_coarse[f]-1 == k for f in fg))
            # (ici on accepte silencieusement — en production, on lèverait un warning)

    if plot:
        fn = feature_names or [str(i) for i in range(shap_vals.shape[1])]
        fig, ax = plt.subplots(figsize=(9, 4))
        dendrogram(Z, labels=fn, ax=ax, color_threshold=0, leaf_rotation=45)
        # Lignes de coupe
        colors_cut = ['#E74C3C', '#3498DB']
        heights = [Z[-(K_coarse-1), 2], Z[-(K_fine-1), 2]]
        for h, c, label in zip(heights, colors_cut,
                                [f"Niveau grossier (K={K_coarse})",
                                 f"Niveau fin (K={K_fine})"]):
            ax.axhline(y=h, color=c, linestyle='--', label=label)
        ax.legend(fontsize=9)
        ax.set_title("Hiérarchie à 2 niveaux découverte depuis les SHAP values")
        plt.tight_layout()
        plt.savefig("figures/auto_hierarchy.png", bbox_inches='tight')
        plt.show()

    return coarse_groups, fine_groups


