"""
Script principal : co-clustering consensus dyadique sur California Housing.
Pipeline : données → SHAP ordre 1 → concaténation VS+ES → triple consensus.

BUGS CORRIGÉS PAR RAPPORT À LA VERSION ORIGINALE :
───────────────────────────────────────────────────
1. [CRITIQUE] Collision de variable `y` : `y` est utilisé pour la target ET
   réécrasé par la latitude dans la boucle géographique.
   → La latitude est maintenant stockée dans `lat` et `lon`.

2. [CRITIQUE] Collision alias `cx` : `import contextily as ctx` vs `cx = x.mean()`
   dans la visualisation géographique. L'alias contextily est renommé `ctx`.

3. [CRITIQUE] Alignement d'index : `row_labels` est un array numpy 0-based
   mais `X` a un index pandas non-contigu après filtrage des outliers.
   → On utilise `.iloc[]` via un index positionnel (`X_reset`).

4. [FONCTIONNEL] Chemin d'import du module coclustering : aligné sur la
   structure du projet (mosaic_shap/clustering/coclustering.py).

5. [AMÉLIORATION] Carte Californie enrichie : enveloppes convexes par cluster,
   centroïdes annotés, fond de carte contextily (optionnel), SHAP moyen
   par cluster affiché en annotation géographique.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
import shap

# ── Import du module coclustering ─────────────────────────────────────────────
# Adapter le chemin selon l'organisation de votre projet :
# Option A (si installé via pip install -e .) :
#   from mosaic_shap.clustering.coclustering import TripleCoclusteringConsensus
# Option B (import direct si fichier local) :
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from coclustering import TripleCoclusteringConsensus

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement et filtrage des données
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ÉTAPE 1 — Chargement des données")
print("=" * 60)

data = fetch_california_housing()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
y_full = pd.Series(data.target, name='MedHouseVal')

# Filtrage outliers (zone Bay Area / San Francisco Bay uniquement)
mask = (
    (X_full['Population'] < 10000) &
    (X_full['AveOccup']   < 6)     &
    (X_full['AveBedrms']  < 1.5)   &
    (X_full['HouseAge']   < 50)    &
    (X_full['Latitude']   .between(37.2, 38.07)) &
    (X_full['Longitude']  .between(-122.5, -121.75))
)
X = X_full.loc[mask].copy()
# BUG CORRIGÉ 1 : `y_target` au lieu de `y` pour éviter l'écrasement en section 5
y_target = y_full.loc[mask].copy()

# Index positionnel 0-based pour aligner avec les labels numpy
# BUG CORRIGÉ 3 : reset_index pour éviter les erreurs d'alignement pandas/numpy
X_reset = X.reset_index(drop=True)
y_reset  = y_target.reset_index(drop=True)

print(f"  N = {len(X_reset)} observations, M = {X_reset.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Entraînement du modèle et calcul SHAP ordre 1
# ══════════════════════════════════════════════════════════════════════════════
print("\nÉTAPE 2 — Modèle + SHAP")
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_reset, y_reset)
print(f"  R² train = {model.score(X_reset, y_reset):.3f}")

explainer  = shap.TreeExplainer(model)
shap_arr   = explainer.shap_values(X_reset)                  # (N, M)
shap_df    = pd.DataFrame(shap_arr, columns=X_reset.columns)  # index 0-based

# ══════════════════════════════════════════════════════════════════════════════
# 3. Matrice dyadique concaténée (VS + ES)
# ══════════════════════════════════════════════════════════════════════════════
print("\nÉTAPE 3 — Construction de la matrice dyadique VS + ES")

def scale_features(X_df: pd.DataFrame, shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise X par sa std, puis pondère par l'importance SHAP moyenne.
    Garantit que les deux espaces (VS et ES) ont des échelles comparables.
    """
    std = X_df.std().replace(0, 1)
    X_norm = (X_df - X_df.mean()) / std
    weight = np.abs(shap_df).mean()     # importance SHAP moyenne par feature
    return X_norm * weight

X_scaled = scale_features(X_reset, shap_df)
X2 = pd.concat([X_scaled, shap_df], axis=1)   # (N, 2×M) — dyadique
print(f"  Matrice X2 : {X2.shape}  ({X_reset.shape[1]} features VS + {shap_df.shape[1]} features ES)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Triple co-clustering consensus
# ══════════════════════════════════════════════════════════════════════════════
print("\nÉTAPE 4 — Triple co-clustering consensus")
print("-" * 40)

coclust = TripleCoclusteringConsensus()
row_labels, col_labels, individual = coclust.compute_consensus(
    X2.values, n_clusters='auto'
)

n_row_clusters = len(np.unique(row_labels))
n_col_clusters = len(np.unique(col_labels))
print(f"\nConsensus final : {n_row_clusters} clusters d'observations, "
      f"{n_col_clusters} clusters de features")

np.save("coclust_row_labels.npy", row_labels)
np.save("coclust_col_labels.npy", col_labels)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\nÉTAPE 5 — Génération des figures")
PALETTE = plt.cm.tab10(np.linspace(0, 1, max(n_row_clusters, 2)))

# ── 5.1 Heatmap réordonnée de X2 ────────────────────────────────────────────
print("  → Fig 1 : heatmap réordonnée...")
order_rows = np.argsort(row_labels)
order_cols = np.argsort(col_labels)
X2_sorted  = X2.iloc[order_rows, order_cols]

# Lignes de séparation entre blocs de clusters de lignes
row_boundaries = []
prev = row_labels[order_rows[0]]
for idx, lab in enumerate(row_labels[order_rows]):
    if lab != prev:
        row_boundaries.append(idx)
        prev = lab

col_boundaries = []
prev = col_labels[order_cols[0]]
for idx, lab in enumerate(col_labels[order_cols]):
    if lab != prev:
        col_boundaries.append(idx)
        prev = lab

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(X2_sorted, cmap='RdBu_r', center=0,
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Valeur normalisée'}, ax=ax)
for rb in row_boundaries:
    ax.axhline(rb, color='black', lw=1.2, alpha=0.8)
for cb in col_boundaries:
    ax.axvline(cb, color='black', lw=1.2, alpha=0.8)
ax.set_title(
    f'Matrice dyadique réordonnée (VS + ES)\n'
    f'{n_row_clusters} clusters d\'observations × {n_col_clusters} clusters de features',
    fontsize=12, fontweight='bold'
)
# Annotation des clusters de lignes
cluster_starts = {}
for pos, lab in enumerate(row_labels[order_rows]):
    if lab not in cluster_starts:
        cluster_starts[lab] = pos
for lab, start in cluster_starts.items():
    end = start
    while end < len(order_rows) - 1 and row_labels[order_rows[end+1]] == lab:
        end += 1
    mid = (start + end) / 2
    ax.text(-0.5, mid, f'C{lab}', ha='right', va='center', fontsize=8,
            color=PALETTE[lab % len(PALETTE)], fontweight='bold')
plt.tight_layout()
plt.savefig('coclust_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("    ✓ coclust_heatmap.png")

# ── 5.2 Heatmap des moyennes par bloc ───────────────────────────────────────
print("  → Fig 2 : moyennes par bloc...")
block_means = np.zeros((n_row_clusters, n_col_clusters))
block_sizes = np.zeros((n_row_clusters, n_col_clusters), dtype=int)
for r in range(n_row_clusters):
    for c in range(n_col_clusters):
        mask_r = (row_labels == r)
        mask_c = (col_labels == c)
        vals = X2.values[np.ix_(mask_r, mask_c)]
        block_means[r, c] = vals.mean()
        block_sizes[r, c] = mask_r.sum()

fig, ax = plt.subplots(figsize=(9, 6))
im = sns.heatmap(
    block_means, annot=True, fmt='.3f', cmap='coolwarm', center=0,
    xticklabels=[f'Col C{i}' for i in range(n_col_clusters)],
    yticklabels=[f'Row C{i}  (n={block_sizes[i,0]})' for i in range(n_row_clusters)],
    ax=ax, linewidths=0.5
)
ax.set_title('Valeur moyenne par bloc (clusters lignes × clusters colonnes)', fontweight='bold')
plt.tight_layout()
plt.savefig('coclust_block_means.png', dpi=150, bbox_inches='tight')
plt.show()
print("    ✓ coclust_block_means.png")

# ── 5.3 Importance SHAP par cluster d'observations ───────────────────────────
print("  → Fig 3 : profil SHAP par cluster...")
feature_names = list(X_reset.columns)
M = len(feature_names)

fig, axes = plt.subplots(1, n_row_clusters, figsize=(5 * n_row_clusters, 5), sharey=True)
if n_row_clusters == 1:
    axes = [axes]

for cluster_id in range(n_row_clusters):
    ax = axes[cluster_id]
    mask_c = (row_labels == cluster_id)
    mean_abs_shap = np.abs(shap_arr[mask_c]).mean(0)
    order = np.argsort(mean_abs_shap)[::-1]
    color = PALETTE[cluster_id % len(PALETTE)]
    bars = ax.barh(
        [feature_names[i] for i in order],
        mean_abs_shap[order],
        color=color, alpha=0.85, edgecolor='white'
    )
    ax.set_title(f'Cluster {cluster_id}\n(n = {mask_c.sum()})',
                 color=color, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|')
    if cluster_id == 0:
        ax.set_ylabel('Feature')
    # Valeur sur barre
    for bar, val in zip(bars, mean_abs_shap[order]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7, color='gray')

fig.suptitle('Profil d\'importance SHAP par cluster d\'observations',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('coclust_shap_profile.png', dpi=150, bbox_inches='tight')
plt.show()
print("    ✓ coclust_shap_profile.png")

# ── 5.4 CARTE CALIFORNIE (version enrichie) ──────────────────────────────────
print("  → Fig 4 : carte Californie enrichie...")

# BUG CORRIGÉ 1 : variables lon/lat distinctes de y_target
# BUG CORRIGÉ 2 : alias contextily renommé `ctx` (pas `cx` qui collision avec centroid_x)
# BUG CORRIGÉ 3 : X_reset (index 0-based) aligné avec row_labels (numpy 0-based)

lon_all = X_reset['Longitude'].values   # (N,) numpy — aligné avec row_labels
lat_all = X_reset['Latitude'].values    # (N,) numpy — aligné avec row_labels
price_all = y_reset.values              # prix médian normalisé pour coloration

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# ── Sous-figure gauche : couleur = cluster d'observation ────────────────────
ax_left = axes[0]

for cluster_id in range(n_row_clusters):
    mask_c = (row_labels == cluster_id)
    lon_c = lon_all[mask_c]
    lat_c = lat_all[mask_c]
    color = PALETTE[cluster_id % len(PALETTE)]

    # Points
    ax_left.scatter(
        lon_c, lat_c,
        c=[color], s=18, alpha=0.65, label=f'Cluster {cluster_id} (n={mask_c.sum()})',
        edgecolors='none', zorder=3
    )

    # Enveloppe convexe
    if mask_c.sum() >= 4:
        pts = np.column_stack((lon_c, lat_c))
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])  # fermer le polygone
            ax_left.fill(hull_pts[:, 0], hull_pts[:, 1],
                         color=color, alpha=0.08, zorder=2)
            ax_left.plot(hull_pts[:, 0], hull_pts[:, 1],
                         color=color, lw=1.8, alpha=0.7, zorder=2)
        except Exception:
            pass

    # Centroïde avec annotation
    centroid_lon = lon_c.mean()
    centroid_lat = lat_c.mean()
    ax_left.scatter(centroid_lon, centroid_lat,
                    marker='*', s=220, c=[color],
                    edgecolors='black', linewidths=1.2, zorder=5)
    ax_left.annotate(
        f'C{cluster_id}\nn={mask_c.sum()}',
        xy=(centroid_lon, centroid_lat),
        xytext=(centroid_lon + 0.04, centroid_lat + 0.03),
        fontsize=9, fontweight='bold', color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color),
        zorder=6
    )

# Fond de carte contextily (optionnel — nécessite `pip install contextily`)
try:
    import contextily as ctx   # BUG CORRIGÉ 2 : alias `ctx` pas `cx`
    ctx.add_basemap(
        ax_left,
        crs='EPSG:4326',
        source=ctx.providers.CartoDB.Positron,
        zoom=10
    )
    print("    → Fond de carte contextily chargé")
except ImportError:
    # Fallback : quadrillage géographique lisible sans carte de fond
    ax_left.set_facecolor('#E8F4F8')
    ax_left.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax_left.set_axisbelow(True)
    print("    → contextily non disponible, fond par défaut utilisé")
    print("       (pip install contextily pour un fond de carte)")

ax_left.set_xlabel('Longitude', fontsize=11)
ax_left.set_ylabel('Latitude', fontsize=11)
ax_left.set_title('Clusters d\'observations — espace dyadique VS+ES\n'
                   '(étoiles = centroïdes, polygones = enveloppes convexes)',
                   fontsize=11, fontweight='bold')
ax_left.legend(loc='upper right', fontsize=9,
               framealpha=0.9, edgecolor='lightgray')

# ── Sous-figure droite : couleur = prix médian, contours = clusters ──────────
ax_right = axes[1]

sc = ax_right.scatter(
    lon_all, lat_all,
    c=price_all, cmap='plasma', s=18, alpha=0.75,
    edgecolors='none', zorder=3, vmin=price_all.min(), vmax=price_all.max()
)
cbar = plt.colorbar(sc, ax=ax_right, label='Prix médian ($100k)', shrink=0.85)

# Contours des clusters superposés (sans remplissage) pour repère spatial
for cluster_id in range(n_row_clusters):
    mask_c = (row_labels == cluster_id)
    lon_c = lon_all[mask_c]
    lat_c = lat_all[mask_c]
    color = PALETTE[cluster_id % len(PALETTE)]

    if mask_c.sum() >= 4:
        pts = np.column_stack((lon_c, lat_c))
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax_right.plot(hull_pts[:, 0], hull_pts[:, 1],
                          color=color, lw=2.2, alpha=0.9,
                          label=f'Cluster {cluster_id}', zorder=4)
        except Exception:
            pass

try:
    import contextily as ctx
    ctx.add_basemap(ax_right, crs='EPSG:4326',
                    source=ctx.providers.CartoDB.Positron, zoom=10)
except ImportError:
    ax_right.set_facecolor('#E8F4F8')
    ax_right.grid(True, linestyle='--', alpha=0.4, color='gray')

ax_right.set_xlabel('Longitude', fontsize=11)
ax_right.set_ylabel('Latitude', fontsize=11)
ax_right.set_title('Prix médian des maisons\n(contours = frontières des clusters dyadiques)',
                    fontsize=11, fontweight='bold')
ax_right.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.suptitle('Co-clustering dyadique VS + ES — Californie (Bay Area)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('coclust_geo_enriched.png', dpi=150, bbox_inches='tight')
plt.show()
print("    ✓ coclust_geo_enriched.png")

# ── 5.5 Carte par cluster : SHAP feature la plus discriminante ───────────────
print("  → Fig 5 : SHAP feature discriminante par cluster...")

# Pour chaque cluster, identifier la feature SHAP la plus discriminante
# = celle dont la valeur moyenne diffère le plus des autres clusters
global_shap_mean = np.abs(shap_arr).mean(0)
discriminance = np.zeros(M)
for cluster_id in range(n_row_clusters):
    mask_c = (row_labels == cluster_id)
    cluster_mean = np.abs(shap_arr[mask_c]).mean(0)
    discriminance += np.abs(cluster_mean - global_shap_mean)
top_feature_idx = int(np.argmax(discriminance))
top_feature     = feature_names[top_feature_idx]
print(f"    Feature la plus discriminante entre clusters : {top_feature}")

fig, ax = plt.subplots(figsize=(10, 8))
shap_vals_top = shap_arr[:, top_feature_idx]
vmax = np.abs(shap_vals_top).max()

sc = ax.scatter(
    lon_all, lat_all,
    c=shap_vals_top, cmap='RdBu_r', s=20, alpha=0.75,
    edgecolors='none', vmin=-vmax, vmax=vmax, zorder=3
)
cbar = plt.colorbar(sc, ax=ax, label=f'SHAP value — {top_feature}', shrink=0.85)

# Contours des clusters
for cluster_id in range(n_row_clusters):
    mask_c = (row_labels == cluster_id)
    lon_c = lon_all[mask_c]
    lat_c = lat_all[mask_c]
    color = PALETTE[cluster_id % len(PALETTE)]
    if mask_c.sum() >= 4:
        pts = np.column_stack((lon_c, lat_c))
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=color, lw=2.5, alpha=0.9, label=f'C{cluster_id}', zorder=4)
            # Annotation SHAP moyen dans le cluster
            mean_shap_cluster = shap_arr[mask_c, top_feature_idx].mean()
            ax.annotate(
                f'C{cluster_id}\nSHAP={mean_shap_cluster:.3f}',
                xy=(lon_c.mean(), lat_c.mean()),
                fontsize=8, fontweight='bold', color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor=color),
                zorder=5
            )
        except Exception:
            pass

try:
    import contextily as ctx
    ctx.add_basemap(ax, crs='EPSG:4326',
                    source=ctx.providers.CartoDB.Positron, zoom=10)
except ImportError:
    ax.set_facecolor('#E8F4F8')
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')

ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title(
    f'SHAP value de "{top_feature}" (feature la plus discriminante entre clusters)\n'
    'Rouge = contribution positive, Bleu = contribution négative',
    fontsize=11, fontweight='bold'
)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig('coclust_geo_shap_discriminant.png', dpi=150, bbox_inches='tight')
plt.show()
print("    ✓ coclust_geo_shap_discriminant.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Résumé textuel
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RÉSUMÉ DES RÉSULTATS")
print("=" * 60)

print(f"\n  N observations   : {len(X_reset)}")
print(f"  M features       : {M}")
print(f"  Clusters lignes  : {n_row_clusters}")
print(f"  Clusters colonnes: {n_col_clusters}")

print("\n  Détail par algorithme individuel :")
for name, res in individual.items():
    n_r = len(np.unique(res['row_labels']))
    n_c = len(np.unique(res['col_labels']))
    print(f"    {name:15s} : {n_r} clusters obs × {n_c} clusters features")

print("\n  Taille des clusters consensuels :")
for cid in range(n_row_clusters):
    n_pts = (row_labels == cid).sum()
    top3_shap = np.argsort(np.abs(shap_arr[row_labels == cid]).mean(0))[::-1][:3]
    top3_names = [feature_names[i] for i in top3_shap]
    print(f"    Cluster {cid} : {n_pts:4d} obs — top SHAP : {', '.join(top3_names)}")

print("\n  Fichiers produits :")
for fname in [
    'coclust_row_labels.npy', 'coclust_col_labels.npy',
    'coclust_heatmap.png', 'coclust_block_means.png',
    'coclust_shap_profile.png', 'coclust_geo_enriched.png',
    'coclust_geo_shap_discriminant.png'
]:
    exists = "✓" if os.path.exists(fname) else "⏳"
    print(f"    {exists}  {fname}")