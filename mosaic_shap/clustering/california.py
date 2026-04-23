"""
Visualisation des clusters sur la carte de Californie.
Fond de carte réel (contextily / Stamen / OpenStreetMap).
Sans enveloppes convexes — uniquement les points colorés par cluster.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
import shap

# Import du module coclustering (adapter selon votre structure)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from coclustering import TripleCoclusteringConsensus

warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10})

# ══════════════════════════════════════════════════════════════════════════════
# 1. Données, modèle, SHAP
# ══════════════════════════════════════════════════════════════════════════════
data = fetch_california_housing()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
y_full = pd.Series(data.target, name='MedHouseVal')

# Filtre Bay Area uniquement
mask = (
    (X_full['Population'] < 10000) &
    (X_full['AveOccup']   < 6)     &
    (X_full['AveBedrms']  < 1.5)   &
    (X_full['HouseAge']   < 50)    &
    (X_full['Latitude'].between(37.2, 38.07)) &
    (X_full['Longitude'].between(-122.5, -121.75))
)
X = X_full.loc[mask].reset_index(drop=True)
y_target = y_full.loc[mask].reset_index(drop=True)
print(f"N = {len(X)} observations")

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y_target)

explainer = shap.TreeExplainer(model)
shap_arr  = explainer.shap_values(X)
shap_df   = pd.DataFrame(shap_arr, columns=X.columns)

# Matrice dyadique VS + ES
std = X.std().replace(0, 1)
X_scaled = (X - X.mean()) / std * np.abs(shap_df).mean()
X2 = pd.concat([X_scaled, shap_df], axis=1)

# Co-clustering
print("Co-clustering consensus...")
coclust = TripleCoclusteringConsensus()
row_labels, col_labels, individual = coclust.compute_consensus(X2.values, n_clusters='auto')
n_clusters = len(np.unique(row_labels))
print(f"{n_clusters} clusters trouvés")

# Coordonnées et prix (alignés avec row_labels via reset_index)
lon      = X['Longitude'].values
lat      = X['Latitude'].values
price    = y_target.values
feature_names = list(X.columns)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARTE CALIFORNIE — points sur fond de carte réel
# ══════════════════════════════════════════════════════════════════════════════

# Palette de couleurs pour les clusters
CMAP_CLUSTERS = plt.cm.tab10
COLORS = [CMAP_CLUSTERS(i / max(n_clusters, 1)) for i in range(n_clusters)]

# ── Tentative de chargement de contextily ────────────────────────────────────
try:
    import contextily as ctx
    HAS_CTX = True
    print("contextily disponible — fond de carte OSM activé")
except ImportError:
    HAS_CTX = False
    print("contextily non disponible — pip install contextily pour le fond de carte")

# ── Tentative d'utilisation de geopandas pour la reprojection ────────────────
try:
    import geopandas as gpd
    from pyproj import Transformer
    HAS_GPD = True
except ImportError:
    HAS_GPD = False


def plot_on_california_map(lon, lat, color_values, labels,
                            title, cmap=None, colorbar_label=None,
                            ax=None, figsize=(12, 10)):
    """
    Trace des points géoréférencés sur un fond de carte Californie.

    Paramètres
    ----------
    lon, lat        : coordonnées WGS84
    color_values    : valeurs pour la coloration (clusters ou SHAP)
    labels          : labels de cluster (pour la légende)
    title           : titre du graphe
    cmap            : colormap (si None → couleurs discrètes par cluster)
    colorbar_label  : label de la colorbar (si cmap continu)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if HAS_CTX:
        # Contextily travaille en Web Mercator (EPSG:3857)
        # → on reprojette les coordonnées WGS84 avant de tracer
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_merc, y_merc = transformer.transform(lon, lat)

        if cmap is not None:
            # Coloration continue (ex : prix, SHAP)
            vmax = np.abs(color_values).max()
            vmin = -vmax if color_values.min() < 0 else color_values.min()
            sc = ax.scatter(x_merc, y_merc, c=color_values, cmap=cmap,
                            s=22, alpha=0.80, edgecolors='none',
                            vmin=vmin, vmax=vmax, zorder=4)
            plt.colorbar(sc, ax=ax, label=colorbar_label, shrink=0.8, pad=0.01)
        else:
            # Coloration discrète par cluster
            unique_labels = np.unique(labels)
            for cid in unique_labels:
                mask_c = (labels == cid)
                ax.scatter(x_merc[mask_c], y_merc[mask_c],
                           c=[COLORS[cid % len(COLORS)]],
                           s=22, alpha=0.80, edgecolors='none',
                           label=f'Cluster {cid}  (n={mask_c.sum()})',
                           zorder=4)

        # Fond de carte OpenStreetMap / CartoDB
        try:
            ctx.add_basemap(ax, crs='EPSG:3857',
                            source=ctx.providers.CartoDB.Positron,
                            zoom='auto', attribution_size=6)
        except Exception:
            try:
                ctx.add_basemap(ax, crs='EPSG:3857',
                                source=ctx.providers.OpenStreetMap.Mapnik,
                                zoom='auto', attribution_size=6)
            except Exception as e:
                print(f"    Fond de carte indisponible : {e}")

        ax.set_xlabel('Longitude (Mercator)', fontsize=10)
        ax.set_ylabel('Latitude (Mercator)', fontsize=10)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(6, 6))

    else:
        # Fallback sans contextily : fond simple avec contours Californie
        _draw_california_background(ax, lon, lat)

        if cmap is not None:
            vmax = np.abs(color_values).max()
            vmin = -vmax if color_values.min() < 0 else color_values.min()
            sc = ax.scatter(lon, lat, c=color_values, cmap=cmap,
                            s=22, alpha=0.85, edgecolors='none',
                            vmin=vmin, vmax=vmax, zorder=4)
            plt.colorbar(sc, ax=ax, label=colorbar_label, shrink=0.8, pad=0.01)
        else:
            unique_labels = np.unique(labels)
            for cid in unique_labels:
                mask_c = (labels == cid)
                ax.scatter(lon[mask_c], lat[mask_c],
                           c=[COLORS[cid % len(COLORS)]],
                           s=22, alpha=0.85, edgecolors='none',
                           label=f'Cluster {cid}  (n={mask_c.sum()})',
                           zorder=4)

        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    return fig, ax


def _draw_california_background(ax, lon_data, lat_data):
    """
    Fond de carte minimal sans dépendance externe :
    - rectangle couleur eau autour de la zone
    - grille géographique lisible
    - marge autour des données
    """
    margin_lon = 0.05
    margin_lat = 0.05
    x0 = lon_data.min() - margin_lon
    x1 = lon_data.max() + margin_lon
    y0 = lat_data.min() - margin_lat
    y1 = lat_data.max() + margin_lat

    # Fond "eau" bleu pâle
    ax.set_facecolor('#D6EAF8')
    # Rectangle "terre" crème
    from matplotlib.patches import FancyBboxPatch
    land = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                           boxstyle="round,pad=0.001",
                           linewidth=1.2, edgecolor='#85929E',
                           facecolor='#F0ECE3', zorder=1)
    ax.add_patch(land)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.grid(True, linestyle='--', alpha=0.35, color='gray', zorder=2)
    ax.set_axisbelow(True)


def add_cluster_centroids(ax, lon, lat, labels, mercator=False):
    """
    Ajoute les centroïdes des clusters avec une étoile et une annotation.
    mercator : si True, reprojette les centroïdes en Web Mercator.
    """
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        mask_c = (labels == cid)
        cx = lon[mask_c].mean()
        cy = lat[mask_c].mean()

        if mercator and HAS_CTX:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            cx, cy = transformer.transform(cx, cy)

        color = COLORS[cid % len(COLORS)]
        ax.scatter(cx, cy, marker='*', s=280, c=[color],
                   edgecolors='black', linewidths=1.0, zorder=6)
        ax.annotate(
            f' C{cid}',
            xy=(cx, cy),
            fontsize=10, fontweight='bold', color=color, zorder=7,
            path_effects=[
                __import__('matplotlib').patheffects.withStroke(
                    linewidth=2.5, foreground='white'
                )
            ]
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Clusters sur fond de carte (la figure principale)
# ══════════════════════════════════════════════════════════════════════════════
print("\n→ Figure 1 : clusters sur carte Californie...")

fig, ax = plt.subplots(figsize=(12, 10))
plot_on_california_map(
    lon=lon, lat=lat,
    color_values=row_labels,
    labels=row_labels,
    title=f'Co-clustering dyadique VS+ES — {n_clusters} clusters\n'
          f'(Bay Area, Californie)',
    ax=ax
)
add_cluster_centroids(ax, lon, lat, row_labels, mercator=HAS_CTX)
ax.legend(loc='lower right', fontsize=9, framealpha=0.92,
          edgecolor='lightgray', markerscale=1.5)

plt.tight_layout()
plt.savefig('map_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ map_clusters.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Prix médian sur fond de carte
# ══════════════════════════════════════════════════════════════════════════════
print("→ Figure 2 : prix médian sur carte Californie...")

fig, ax = plt.subplots(figsize=(12, 10))
plot_on_california_map(
    lon=lon, lat=lat,
    color_values=price,
    labels=row_labels,
    title='Prix médian des maisons — Bay Area\n(en centaines de milliers de $)',
    cmap='plasma',
    colorbar_label='Prix médian ($100k)',
    ax=ax
)
add_cluster_centroids(ax, lon, lat, row_labels, mercator=HAS_CTX)

plt.tight_layout()
plt.savefig('map_price.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ map_price.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — SHAP de la feature la plus discriminante
# ══════════════════════════════════════════════════════════════════════════════
print("→ Figure 3 : SHAP feature discriminante sur carte...")

global_mean = np.abs(shap_arr).mean(0)
disc = np.zeros(len(feature_names))
for cid in np.unique(row_labels):
    disc += np.abs(np.abs(shap_arr[row_labels == cid]).mean(0) - global_mean)
top_idx     = int(np.argmax(disc))
top_feature = feature_names[top_idx]
shap_top    = shap_arr[:, top_idx]
print(f"  Feature la plus discriminante : {top_feature}")

fig, ax = plt.subplots(figsize=(12, 10))
plot_on_california_map(
    lon=lon, lat=lat,
    color_values=shap_top,
    labels=row_labels,
    title=f'Valeur SHAP de "{top_feature}" — feature la plus discriminante entre clusters\n'
          '(rouge = contribution positive, bleu = contribution négative)',
    cmap='RdBu_r',
    colorbar_label=f'SHAP — {top_feature}',
    ax=ax
)

plt.tight_layout()
plt.savefig('map_shap_discriminant.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ map_shap_discriminant.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Grille : une sous-carte par cluster, couleur = prix
# ══════════════════════════════════════════════════════════════════════════════
print("→ Figure 4 : grille une sous-carte par cluster...")

ncols = min(n_clusters, 3)
nrows = int(np.ceil(n_clusters / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(6 * ncols, 5.5 * nrows),
                          squeeze=False)

vmin_p, vmax_p = price.min(), price.max()

for cid in range(n_clusters):
    row_idx = cid // ncols
    col_idx = cid % ncols
    ax = axes[row_idx][col_idx]
    mask_c = (row_labels == cid)

    _draw_california_background(ax, lon, lat)

    # Tous les points en gris clair (contexte)
    ax.scatter(lon[~mask_c], lat[~mask_c],
               c='#CCCCCC', s=8, alpha=0.35, edgecolors='none', zorder=3)

    # Points du cluster en avant-plan, colorés par prix
    sc = ax.scatter(lon[mask_c], lat[mask_c],
                    c=price[mask_c], cmap='plasma',
                    s=28, alpha=0.90, edgecolors='none',
                    vmin=vmin_p, vmax=vmax_p, zorder=4)

    color = COLORS[cid % len(COLORS)]
    ax.set_title(f'Cluster {cid}  (n = {mask_c.sum()})',
                  color=color, fontweight='bold', fontsize=11)
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)

    # Top 3 features SHAP du cluster
    top3 = np.argsort(np.abs(shap_arr[mask_c]).mean(0))[::-1][:3]
    top3_str = ', '.join(feature_names[i] for i in top3)
    ax.text(0.02, 0.03, f'Top SHAP : {top3_str}',
             transform=ax.transAxes, fontsize=7.5, color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    plt.colorbar(sc, ax=ax, label='Prix ($100k)', shrink=0.85, pad=0.01)

# Masquer les axes en surplus
for idx in range(n_clusters, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.suptitle('Vue par cluster : prix médian sur la zone Bay Area\n'
             '(points gris = autres clusters)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('map_grid_per_cluster.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ map_grid_per_cluster.png")

print("\nFigures générées :")
for f in ['map_clusters.png', 'map_price.png',
          'map_shap_discriminant.png', 'map_grid_per_cluster.png']:
    print(f"  ✓ {f}")