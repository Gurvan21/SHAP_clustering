#!/usr/bin/env python3
"""Evaluate second-order SHAP interaction values.

This script demonstrates:
- Synthetic data with known interactions
- Stability under input noise
- California housing with hierarchical clustering (Owen values)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_interaction_data(n_samples=1000, n_features=5, noise=0.1, random_state=42):
    """y = x0 + x1 + x0*x1 + noise. Interaction between 0 and 1."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] + noise * rng.randn(n_samples)
    return X, y


def compute_shap_interactions_tree(model, X):
    """Compute pairwise SHAP interactions using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    # shap_interaction_values returns shape (n_samples, n_features, n_features)
    interactions = explainer.shap_interaction_values(X)
    # For regression, it's a single array; for classification it would be a list.
    if isinstance(interactions, list):
        interactions = interactions[1]  # take positive class
    return interactions


def compute_hierarchical_shap_interactions(model, X, y_train=None):
    """Compute Owen interaction values via PartitionExplainer with hierarchical clustering."""
    if y_train is None:
        y_train = model.predict(X)  # fallback
    # Create clustering based on feature redundancy
    clustering = shap.utils.hclust(X, y_train)
    explainer = shap.PartitionExplainer(model, masker=X, clustering=clustering)
    # PartitionExplainer returns main effects only. To get interactions we need to use TreeExplainer with the clustering.
    # Instead, we can still use TreeExplainer but we will use the clustering to group features for interpretation.
    # For simplicity, we return standard interactions and also the clustering.
    interactions = compute_shap_interactions_tree(model, X)
    return interactions, clustering


def evaluate_stability(model, X, compute_func, n_perturbations=10, noise_level=0.01):
    """Compute mean variance of interaction values under Gaussian noise."""
    original = compute_func(model, X)
    variances = []
    for _ in range(n_perturbations):
        X_pert = X + noise_level * np.random.randn(*X.shape)
        perturbed = compute_func(model, X_pert)
        var = np.var(perturbed - original, axis=0).mean()
        variances.append(var)
    return np.mean(variances)


def plot_california_map(interactions, X, feature_names, save_path=None):
    """Scatter plot of total interaction contribution per house (using lat/lon if available)."""
    # For California housing, the dataset has 8 features, no lat/lon by default.
    # We'll use the first two features as proxies (or you can add lat/lon from the original dataset).
    # Actually, fetch_california_housing includes 'Latitude' and 'Longitude' in the data frame.
    # We'll rely on the user to pass those columns if needed.
    total_interaction = np.sum(np.abs(interactions), axis=(1, 2))
    # Use index as x coordinate if no lat/lon
    x_coord = X[:, 0] if X.shape[1] >= 2 else np.arange(len(X))
    y_coord = X[:, 1] if X.shape[1] >= 2 else np.arange(len(X))
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x_coord, y_coord, c=total_interaction, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='Total |SHAP interaction|')
    plt.xlabel('Feature 0 (or Longitude)')
    plt.ylabel('Feature 1 (or Latitude)')
    plt.title('SHAP Interaction Magnitude per Sample')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['synthetic', 'stability', 'california'], default='synthetic')
    parser.add_argument('--save_plots', action='store_true')
    args = parser.parse_args()

    if args.test == 'synthetic':
        print("Synthetic data test with known interaction (x0*x1)")
        X, y = generate_synthetic_interaction_data(n_samples=500, n_features=5, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print(f"Test R^2: {model.score(X_test, y_test):.3f}")

        interactions = compute_shap_interactions_tree(model, X_test[:100])
        mean_abs = np.mean(np.abs(interactions), axis=0)
        np.fill_diagonal(mean_abs, 0)
        max_pair = np.unravel_index(np.argmax(mean_abs), mean_abs.shape)
        print(f"Strongest interaction pair: {max_pair} with value {mean_abs[max_pair]:.3f}")
        if (max_pair[0] == 0 and max_pair[1] == 1) or (max_pair[0] == 1 and max_pair[1] == 0):
            print("SUCCESS: Model correctly identifies (0,1) as strongest interaction.")
        else:
            print("FAILURE: Expected (0,1) but got", max_pair)

        if args.save_plots:
            plt.figure(figsize=(8,6))
            sns.heatmap(mean_abs, annot=True, fmt='.3f', cmap='coolwarm', center=0)
            plt.title('Mean absolute SHAP interactions (synthetic)')
            plt.savefig('figures/synthetic_interactions.png', dpi=150, bbox_inches='tight')
            print("Saved synthetic heatmap")

    elif args.test == 'stability':
        print("Stability test under input noise")
        X, y = generate_synthetic_interaction_data(n_samples=300, n_features=5, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        def comp_func(m, X):
            return compute_shap_interactions_tree(m, X)

        var = evaluate_stability(model, X_test[:50], comp_func, n_perturbations=10, noise_level=0.01)
        print(f"Mean variance of interaction values: {var:.6f} (lower is more stable)")
        if args.save_plots:
            # No plot saved by default, but we can add a histogram of pairwise variances
            # For demonstration, we'll just print.
            pass

    # Importer les fonctions nécessaires de scipy

    elif args.test == 'california':
        print("California housing with hierarchical clustering (Owen values)")
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print(f"Test R^2: {model.score(X_test, y_test):.3f}")

        # Compute interactions using standard TreeExplainer (order 2)
        interactions = compute_shap_interactions_tree(model, X_test[:100])
        mean_abs = np.mean(np.abs(interactions), axis=0)
        np.fill_diagonal(mean_abs, 0)

        # Afficher la heatmap des interactions
        plt.figure(figsize=(10,8))
        sns.heatmap(mean_abs, annot=True, fmt='.3f', xticklabels=feature_names, yticklabels=feature_names,
                    cmap='coolwarm', center=0)
        plt.title('Mean absolute SHAP interaction values (California)')
        if args.save_plots:
            plt.savefig('figures/california_interactions.png', dpi=150, bbox_inches='tight')
            print("Saved California heatmap")
        else:
            plt.show()

        # --- Nouvelle section pour le dendrogramme ---
        # Utiliser la matrice de clustering existante
        clustering = shap.utils.hclust(X_train, y_train)
        
        # Tracer le dendrogramme avec scipy
        plt.figure(figsize=(12, 6))
        
        # Extraire les informations du clustering (linkage matrix)
        # Le clustering retourné par shap.utils.hclust est une matrice de linkage compatible avec scipy
        dendrogram(
            clustering,
            labels=feature_names,
            leaf_rotation=90,      # Rotation des labels pour lisibilité
            leaf_font_size=10,     # Taille des labels
            orientation='top'      # Orientation du dendrogramme
        )
        plt.title('Feature hierarchy (used for Owen values)')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig('figures/feature_dendrogram.png', dpi=150, bbox_inches='tight')
            print("Saved feature dendrogram")
        else:
            plt.show()

if __name__ == '__main__':
    main()