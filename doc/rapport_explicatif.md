# Segmentation hiérarchique causale du modèle California Housing

## 1. Objectif général

L’objectif de ce travail est de **comprendre un modèle de régression (Gradient Boosting) sur California Housing** à travers deux axes :

- **Axe vertical (variables)** : comment les variables se structurent en groupes, interagissent et se partagent le pouvoir explicatif (SHAP, Owen, Coop/Comp, futur Winter).
- **Axe horizontal (observations)** : comment l’espace des points se segmente en **régions locales** où le modèle se comporte de façon homogène (clusters ordre 1 / ordre 2, DAGs locaux).

À terme, l’idée est de se rapprocher d’une **segmentation hiérarchique causale** : des régions de données (clusters) avec, pour chacune, une petite hiérarchie de variables importante localement, et un futur lien avec les **valeurs de Winter hiérarchiques**.

---

## 2. Clustering hiérarchique dans l’espace des explications

### 2.1 SHAP ordre 1 et ordre 2

- J’ai d’abord entraîné un **GradientBoostingRegressor** sur 3000 points California Housing.
- Avec un **Fast TreeExplainer** (TreeSHAP optimisé pour les arbres), j’ai calculé :
  - des **valeurs SHAP d’ordre 1** (effet individuel de chaque feature),
  - des **interactions d’ordre 2** (SHAP interaction values) sur des sous-ensembles.

Ces valeurs shapley me donnent une première « carte » de **qui explique quoi** au niveau des variables, indépendamment de toute structure causale.

### 2.2 Clustering ordre 1 et ordre 2 (dimension horizontale)

- À partir des SHAP ordre 1, j’ai projeté les explications en 2D (UMAP) puis appliqué **HDBSCAN** :
  - cela donne les **clusters ordre 1** (8 groupes) que l’on voit dans le rapport (heatmap cluster × SHAP, UMAP, cartes).
- Pour chaque cluster ordre 1, j’ai ensuite recalculé des **SHAP d’ordre 2** et reclustérisé :
  - cela donne les **sous-clusters ordre 2 par cluster** (dossier `california_order2_per_order1_cluster`),
  - avec leurs propres heatmaps, UMAP et cartes.

On obtient ainsi une **segmentation hiérarchique horizontale** : d’abord de grands types de comportements SHAP, puis à l’intérieur de chaque type, des sous-comportements plus fins.

---

## 3. Approche causale : global vs local

### 3.1 DAG global et do-Shapley

- En parallèle, j’ai mis en place un pipeline **causal global** :
  - découverte d’un **DAG sur les variables** (méthode GES/PC via causal-learn),
  - estimation d’un SCM (DoWhy GCM),
  - calcul de **do-Shapley** : attributions inspirées de Shapley mais basées sur des **interventions do(\(\cdot\))**.
- Les résultats sont visibles dans `figures/causal_shap_california` :
  - graphe causal global, heatmap cluster × do-Shapley, UMAP, cartes.

Ici, la différence majeure avec SHAP classique est que les contributions sont censées refléter des **effets plus proches du causal** (en tenant compte du DAG), et pas seulement des corrélations.

### 3.2 DAGs locaux par cluster (exploration causale locale)

Pour rapprocher la **dimension horizontale** (clusters) de la causalité, j’ai ensuite fait une **exploration causale locale** :

- Pour chaque cluster ordre 1 :
  - je garde les **top‑k features** les plus importantes (|SHAP| moyen),
  - je lance à nouveau la **découverte de DAG** (GES) uniquement sur ces variables + `price`,
  - je sauvegarde le graphe causal local (`causal_shap_california_per_order1_cluster/cluster_k/`).

Ces DAGs locaux permettent de voir **comment les mécanismes causaux se spécialisent par cluster** :
par exemple, certains clusters où le bloc spatial domine, d’autres où le bloc socio‑économique est central, etc.

---

## 4. Groupes de variables (Owen-like) et indices Coop/Comp

### 4.1 Blocs de variables (inspirés d’Owen/Winter)

Pour commencer à structurer l’axe vertical, j’ai défini 4 **blocs de variables** cohérents (dans l’esprit des groupes Owen/Winter) :

- `spatial` = {Latitude, Longitude}
- `socio_eco` = {MedInc, HouseAge}
- `stock` = {AveRooms, AveBedrms}
- `density` = {Population, AveOccup}

Ces blocs sont utilisés dans plusieurs scripts :

- `run_owen_winter_per_cluster.py` : agrège les SHAP ordre 1 **par bloc** pour chaque cluster, et produit des barplots d’importance par feature et par coalition.
- `plot_coalition_views.py` : produit une **heatmap clusters × coalitions**, un **UMAP couleur = coalition dominante**, et une **carte Californie couleur = coalition dominante**.

Cela donne une première vision « Owen‑like » : **quel bloc de variables domine dans quel cluster**.

### 4.2 Indices de coopération / compétition à l’intérieur des blocs

Pour aller plus loin à l’intérieur des blocs, j’ai introduit des **indices de coopération / compétition** :

- Pour une paire de variables (i, j) dans un bloc et un point x :
  - je regarde la décision binaire du modèle (par ex. prix au-dessus vs en dessous de la médiane) avec :
    - ni i ni j (baseline),
    - seulement i,
    - seulement j,
    - i et j ensemble.
- **Coopération** :
  - quand ni i ni j seules ne changent la décision,
  - mais i + j ensemble la changent.
- **Compétition** :
  - quand au moins une seule des deux suffit à changer la décision.

Le script `run_coop_competition_per_block.py` :

- calcule, pour chaque bloc et chaque cluster, la proportion d’événements de **coopération** vs **compétition**,
- produit des **graphes Coop/Comp par coalition** (dossier `coop_comp_california_per_order1_cluster`).

Ces graphes indiquent si, **à l’intérieur d’un bloc**, les variables ont plutôt un rôle **complémentaire (coopération)** ou **substituable (compétition)**.

---

## 5. Préoccupations et limites actuelles

### 5.1 Taille d’échantillon et do-calculus local

Une des inquiétudes est que **3000 points**, une fois répartis sur 8 clusters ordre 1 puis sur des sous-clusters ordre 2, **ne suffisent peut‑être pas** pour :

- estimer de manière robuste des **DAG locaux** (GES peut devenir instable avec peu de points),
- faire du **do‑calculus local** (estimations P(price | do(X=x)) fiables) dans chaque région.

Concrètement :

- Les DAGs locaux doivent être interprétés comme des **hypothèses exploratoires** plutôt que comme des vérités fortes.
- Les futures expériences avec pyAgrum (réseaux bayésiens + do(\(\cdot\))) gagneraient à être faites sur :
  - plus de données,
  - ou des clusters plus gros / moins nombreux,
  - et avec des tests de robustesse (bootstrap, score BIC, etc.).

### 5.2 Construction des groupes Owen/Winter : corrélation vs graphe causal

Pour construire les blocs de variables (Owen/Winter hiérarchique), j’hésite entre :

- **Heatmap de corrélation / similarité SHAP** :
  - simple à mettre en œuvre (distance 1 − |corr(φᵢ, φⱼ)|),
  - donne une hiérarchie de features alignée avec la **similarité dans l’espace des explications**.
- **Graphe causal** (DAG global ou local) :
  - plus ambitieux : utiliser la structure du DAG pour regrouper les variables (par ex. voisins proches, sous‑graphes denses),
  - mais plus délicat, car le DAG appris peut être fragile avec peu de données, et mélanger des liens causaux et des liens résiduels.

Pour l’instant, j’ai **fixé une partition raisonnable à la main** (spatial, socio‑éco, stock, densité), en m’appuyant sur la sémantique et sur le DAG, mais l’étape importante à venir sera de **découvrir cette hiérarchie automatiquement** (clustering hiérarchique de corrélations SHAP, puis enrichissement avec le DAG).

---

## 6. Lien avec le workflow Winter / Shapley Causal que j’ai lu

J’ai lu ton workflow en 6 étapes (SHAP, interactions, parcellisation, tessellation, personas, narration), qui sert de **référence** :

- **Étape 0 – Données & modèle**  
  → Alignée : j’ai un dataset (California Housing) et un modèle ML (Gradient Boosting) que je cherche à « ouvrir ».

- **Étape 1 – Valeurs de Shapley & indices**  
  → Alignée : j’ai calculé des Shapley d’ordre 1 (SHAP), mis en place des blocs Owen‑like, et commencé des indices plus complexes (Coop/Comp) à l’intérieur des blocs.
  → C’est une première étape vers des **indices hiérarchiques type Winter** (avec une structure de groupes simple pour l’instant).

- **Étape 2 – Interactions**  
  → Alignée : j’utilise les SHAP d’ordre 2 (interactions) pour comprendre les effets conjoints et pour explorer l’intérieur des blocs.

- **Étape 3 – Parcellisation**  
  → Alignée : les clusters ordre 1 et ordre 2 jouent le rôle de **parcelles** dans l’espace des explications (UMAP + HDBSCAN).

- **Étapes 4–5 – Tessellation & Personas**  
  → Encore partiellement à faire : je n’ai pas encore remplacé le modèle global par des modèles locaux interprétables ni créé des « personas » textuels, mais les DAGs locaux et les blocs coop/comp vont dans cette direction (décrire un « type de région » et ce qui la fait changer).

- **Étape 6+ – Narration globale, biais, monitoring**  
  → Ce sera l’étape suivante : utiliser toutes ces briques (clusters, DAGs locaux, blocs Owen, Coop/Comp, futur Winter) pour raconter une histoire cohérente sur « comment le modèle fonctionne » et comment il varie régionalement.

En résumé, mon travail actuel se situe principalement aux **Étapes 1–3** du workflow, avec des expérimentations causales (DAG global/local, do‑Shapley) qui préparent le terrain pour :

- une future **hiérarchie Winter causale** (Winter avec un jeu défini par des espérances interventionnelles do(\(\cdot\))),
- et une meilleure articulation entre **structure verticale** (hiérarchie de variables) et **structure horizontale** (parcelles / clusters locaux).

---

## 7. Conclusion

Ce projet est un **premier essai structuré** pour :

- combiner **clustering hiérarchique des explications** (ordre 1 / ordre 2),
- avec une **exploration causale globale et locale** (DAG, do‑Shapley),
- et des **blocs de variables** inspirés d’Owen/Winter, enrichis par des indices de coopération / compétition.

Les résultats doivent encore être consolidés (taille d’échantillon, robustesse des DAGs, construction automatique de la hiérarchie), mais ils constituent une base pour :

- mieux appréhender le comportement du modèle,
- préparer l’application de **valeurs de Winter hiérarchiques causales**,
- et, à terme, construire un véritable **workflow de segmentation hiérarchique causale** conforme au schéma que tu proposes.

