# Documentation – Étude clusters California Housing

## Rapport principal

Ouvrir le rapport HTML dans un navigateur pour consulter toutes les figures avec légendes et interprétations :

- **Fichier :** [rapport_clusters_california.html](rapport_clusters_california.html)
- **Emplacement :** depuis la racine du projet : `doc/rapport_clusters_california.html`

Les images sont chargées depuis `../figures/`. Il faut donc ouvrir le fichier depuis le projet (par exemple en double-cliquant sur le fichier ou en ouvrant depuis le répertoire du projet) pour que les chemins relatifs fonctionnent.

### Contenu du rapport

1. **Vue d’ensemble ordre 1 (SHAP)** – Résumé des 8 clusters (effectif, prix moyen), heatmap cluster vs SHAP, UMAP (cluster et prix), cartes Californie.
2. **Exploration causale globale (do-Shapley)** – DAG découvert, heatmap do-Shapley, UMAP, cartes.
3. **Détail par cluster** – Pour chaque cluster 0–7, un **seul bloc** regroupe : infos (effectif, prix moyen, top features, arêtes DAG), **graphe causal local**, puis sous-clusters ordre 2 (heatmap si disponible, UMAP cluster/prix, cartes cluster/prix). Toutes les figures liées à un cluster sont ainsi regroupées pour faciliter l’observation et l’interprétation.
