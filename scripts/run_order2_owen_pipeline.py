"""
Pipeline complète avec visualisations comparatives
Shapley ordre 1 vs Owen vs Winter vs Interactions ordre 2
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
#import matplotlib
#matplotlib.use('Agg')   # backend non interactif, sans limitation mémoire
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns


from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kendalltau
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

from mosaic_shap.data.synthetic import make_dataset_overlap_scores_but_separable_interactions
from mosaic_shap.data.Data_Pedagogique import make_dataset_Housing_California
from mosaic_shap.explain import (
    Order2TreeSHAPInteractions, Order1TreeSHAP,
    OWENExplainer, WINTERExplainer
)
from mosaic_shap.explain.Owen_Shap.grouping import discover_two_level_hierarchy
from mosaic_shap.pipeline.vectorize import vectorize_interactions

os.makedirs("figures", exist_ok=True)

# ── Style global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
})
BLUE   = "#2980B9"
ORANGE = "#E67E22"
PURPLE = "#8E44AD"
GREEN  = "#27AE60"
RED    = "#E74C3C"
COLORS = [BLUE, ORANGE, PURPLE, GREEN, RED, "#F39C12", "#16A085", "#8E44AD"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DONNÉES & MODÈLE
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ÉTAPE 1 — Données & Modèle")
print("=" * 60)

n, seed = 10000, 42
#X, y, _ = make_dataset_overlap_scores_but_separable_interactions(n=n, seed=seed)
X, y, meta = make_dataset_Housing_California(n=n, seed_=seed)

model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, min_samples_split=5, learning_rate=0.05, random_state=seed
    ).fit(X, y)
background = X[:50]
M = X.shape[1]
#feature_names = [f"x{i}" for i in range(M)]
feature_names = list(meta["feature_names"])

print(f"  N={n} observations, M={M} features")
#print(f"  Classes : {np.bincount(y)}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. CALCUL DES EXPLICABILITÉS
# ═══════════════════════════════════════════════════════════════════════════
print("\nÉTAPE 2 — Calcul des explicabilités")

print("  → Shapley ordre 1...")
shap1 = Order1TreeSHAP().compute(model, X)             # (N, M)

print("  → Shapley ordre 2 (interactions)...")
shap2 = Order2TreeSHAPInteractions().compute(model, X).values     # (N, M, M)

# Hiérarchie pour Owen/Winter
try:
    coarse_groups, fine_groups = discover_two_level_hierarchy(
        shap1, K_coarse=2, K_fine=min(4, M), plot=False
    )
    all_fine = sorted([f for g in fine_groups for f in g])
    assert all_fine == list(range(M))
    print(f"  → Hiérarchie : {len(coarse_groups)} groupes grossiers, {len(fine_groups)} fins")
except Exception:
    fine_groups   = [list(range(M//2)), list(range(M//2, M))]
    coarse_groups = [list(range(M))]
    print("  → Hiérarchie : fallback groupes égaux")

N_EXPLAIN  = 80
N_PERM     = 16

print(f"  → Owen (N={N_EXPLAIN}, n_perm={N_PERM})...")
owen_vals = OWENExplainer(
    model, fine_groups, background,
    n_permutations=N_PERM, feature_names=None
).shap_values(X[:N_EXPLAIN])

print(f"  → Winter (N={N_EXPLAIN}, n_perm={N_PERM})...")
winter_vals = WINTERExplainer(
    model, coarse_groups, fine_groups, background,
    n_permutations=N_PERM, feature_names=None
).shap_values(X[:N_EXPLAIN])

# Clustering sur ordre 2
Z, pair_idx = vectorize_interactions(shap2, include_diag=False)
Zp     = PCA(n_components=min(10, Z.shape[1])).fit_transform(Z)
labels = HDBSCAN(min_cluster_size=20).fit_predict(Zp)
n_clusters = len(np.unique(labels[labels >= 0]))
print(f"  → Clustering : {n_clusters} clusters HDBSCAN")

# Taille commune pour comparaisons
N_COMMON = min(len(shap1), len(owen_vals), len(winter_vals))
s1 = shap1[:N_COMMON]
ow = owen_vals[:N_COMMON]
wi = winter_vals[:N_COMMON]


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Importance moyenne : Shapley vs Owen vs Winter
# ═══════════════════════════════════════════════════════════════════════════
print("\nÉTAPE 3 — Génération des figures")
print("  → Figure 1 : Importances comparées...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Importance moyenne par feature\n(Mean |valeur| pour chaque méthode)",
             fontsize=13, fontweight='bold', y=1.02)

methods = [("Shapley ordre 1", s1, BLUE),
           ("Owen",            ow, ORANGE),
           ("Winter",          wi, PURPLE)]

base_order = np.argsort(np.abs(s1).mean(0))[::-1]

for ax, (name, vals, color) in zip(axes, methods):
    imp = np.abs(vals).mean(0)
    bars = ax.bar(
        [feature_names[i] for i in base_order],
        imp[base_order],
        color=color, alpha=0.85, edgecolor='white', linewidth=0.8
    )
    # Valeur sur chaque barre
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                f"{h:.3f}", ha='center', va='bottom', fontsize=7, color='gray')
    ax.set_title(name, color=color)
    ax.set_ylabel("Mean |valeur|" if ax == axes[0] else "")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_ylim(0, np.abs(s1).mean(0).max() * 1.25)

plt.tight_layout()
plt.savefig("figures/fig1_importance_comparison.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig1_importance_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scatter croisés : Shapley vs Owen vs Winter
# ═══════════════════════════════════════════════════════════════════════════
print("  → Figure 2 : Scatter croisés...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Corrélation des attributions individuelles\n(chaque point = une observation × feature)",
             fontsize=12, fontweight='bold')

for ax, (name, vals, color) in zip(axes, [("Owen", ow, ORANGE), ("Winter", wi, PURPLE)]):
    sv = s1.ravel()
    ov = vals.ravel()
    r  = np.corrcoef(sv, ov)[0, 1]

    # Hexbin pour gérer la densité
    hb = ax.hexbin(sv, ov, gridsize=25, cmap='YlOrRd', mincnt=1, linewidths=0.2)
    plt.colorbar(hb, ax=ax, label="Nombre de points")

    lim = max(np.abs(sv).max(), np.abs(ov).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, alpha=0.6, label="y=x")
    ax.set_xlabel("Shapley ordre 1", fontsize=10)
    ax.set_ylabel(name, color=color, fontsize=10)
    ax.set_title(f"Shapley vs {name}  (r = {r:.3f})", color=color)
    ax.legend(fontsize=8)

    # Annotation r
    ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("figures/fig2_scatter_cross.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig2_scatter_cross.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Kendall τ entre toutes les méthodes
# ═══════════════════════════════════════════════════════════════════════════
print("  → Figure 3 : Matrice de Kendall τ...")

method_names = ["Shapley", "Owen", "Winter"]
method_imps  = [np.abs(s1).mean(0), np.abs(ow).mean(0), np.abs(wi).mean(0)]

tau_mat = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        tau_mat[i, j], _ = kendalltau(method_imps[i], method_imps[j])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cohérence entre méthodes d'explicabilité", fontsize=13, fontweight='bold')

# Heatmap Kendall τ
ax = axes[0]
mask = np.eye(3, dtype=bool)
cmap = LinearSegmentedColormap.from_list("rg", ["#E74C3C", "#F39C12", "#27AE60"])
im = ax.imshow(tau_mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(method_names)
ax.set_yticks(range(3)); ax.set_yticklabels(method_names)
ax.set_title("Kendall τ des rankings d'importance\n(1 = même ranking, 0 = indépendant)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{tau_mat[i,j]:.2f}", ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if tau_mat[i,j] < 0.5 else 'black')
plt.colorbar(im, ax=ax)

# Divergence par feature : |Owen - Shapley| et |Winter - Shapley|
ax = axes[1]
div_owen   = np.abs(ow - s1).mean(0)
div_winter = np.abs(wi - s1).mean(0)
x_pos = np.arange(M)
w = 0.35
b1 = ax.bar(x_pos - w/2, div_owen,   w, label="Owen - Shapley",   color=ORANGE, alpha=0.85)
b2 = ax.bar(x_pos + w/2, div_winter, w, label="Winter - Shapley", color=PURPLE, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel("Mean |méthode - Shapley|")
ax.set_title("Divergence par feature\n(signal de structure de groupe)")
ax.legend(fontsize=9)
ax.axhline(div_owen.mean(),   color=ORANGE, ls='--', lw=1.2, alpha=0.7)
ax.axhline(div_winter.mean(), color=PURPLE, ls='--', lw=1.2, alpha=0.7)

plt.tight_layout()
plt.savefig("figures/fig3_consistency.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig3_consistency.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Distribution des valeurs pour chaque feature
# ═══════════════════════════════════════════════════════════════════════════
print("  → Figure 4 : Distributions par feature...")

n_rows = 2
n_cols = (M + 1) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
axes = axes.flatten()
fig.suptitle("Distribution des attributions par feature\n(Shapley / Owen / Winter)",
             fontsize=13, fontweight='bold')

for i in range(M):
    ax = axes[i]
    for vals, name, color, alpha in [
        (s1, "Shapley", BLUE,   0.6),
        (ow, "Owen",    ORANGE, 0.6),
        (wi, "Winter",  PURPLE, 0.6),
    ]:
        ax.hist(vals[:, i], bins=18, alpha=alpha, color=color,
                label=name, density=True, edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(feature_names[i], fontsize=10)
    ax.set_xlabel("Valeur")
    if i % n_cols == 0:
        ax.set_ylabel("Densité")
    if i == 0:
        ax.legend(fontsize=7)

# Masquer les axes en trop
for j in range(M, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("figures/fig4_distributions.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig4_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Interactions ordre 2 : heatmap globale & par classe
# ═══════════════════════════════════════════════════════════════════════════
"""print("  → Figure 5 : Interactions ordre 2...")

n_cols_fig = 2 + len(np.unique(y))
fig, axes = plt.subplots(1, n_cols_fig, figsize=(n_cols_fig * 4, 4.5))
fig.suptitle("Matrices d'interactions Shapley ordre 2\n(Mean |φᵢⱼ|)",
             fontsize=13, fontweight='bold')

cmap_inter = LinearSegmentedColormap.from_list("inter", ["white", "#2980B9", "#1A252F"])
vmax = np.abs(shap2).mean(0).max()

# Globale
ax = axes[0]
mat_global = np.abs(shap2).mean(0)
im = ax.imshow(mat_global, cmap=cmap_inter, vmin=0, vmax=vmax)
ax.set_xticks(range(M)); ax.set_xticklabels(feature_names, rotation=45, fontsize=8)
ax.set_yticks(range(M)); ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title("Global (toutes classes)", fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# Par classe y=0
ax = axes[1]
mat_y0 = np.abs(shap2[y == 0]).mean(0)
im = ax.imshow(mat_y0, cmap=cmap_inter, vmin=0, vmax=vmax)
ax.set_xticks(range(M)); ax.set_xticklabels(feature_names, rotation=45, fontsize=8)
ax.set_yticks(range(M)); ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title("Classe y=0", fontweight='bold', color=BLUE)
plt.colorbar(im, ax=ax, shrink=0.8)

# Par classe y=1
ax = axes[2]
mat_y1 = np.abs(shap2[y == 1]).mean(0)
im = ax.imshow(mat_y1, cmap=cmap_inter, vmin=0, vmax=vmax)
ax.set_xticks(range(M)); ax.set_xticklabels(feature_names, rotation=45, fontsize=8)
ax.set_yticks(range(M)); ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title("Classe y=1", fontweight='bold', color=ORANGE)
plt.colorbar(im, ax=ax, shrink=0.8)

# Différentiel y1 - y0
if n_cols_fig > 3:
    ax = axes[3]
    diff = mat_y1 - mat_y0
    lim  = np.abs(diff).max()
    im2  = ax.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax.set_xticks(range(M)); ax.set_xticklabels(feature_names, rotation=45, fontsize=8)
    ax.set_yticks(range(M)); ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("Différentiel (y1 - y0)", fontweight='bold', color=RED)
    plt.colorbar(im2, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("figures/fig5_interactions_order2.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig5_interactions_order2.png")
"""

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Top interactions par cluster HDBSCAN
# ═══════════════════════════════════════════════════════════════════════════
print("  → Figure 6 : Top interactions par cluster...")

unique_clusters = np.unique(labels[labels >= 0])

if len(unique_clusters) == 0:
    print("    ⚠️  Pas de clusters — figure ignorée")
else:
    fig, axes = plt.subplots(1, len(unique_clusters), figsize=(len(unique_clusters) * 5, 4.5))
    if len(unique_clusters) == 1:
        axes = [axes]
    fig.suptitle("Top interactions par cluster HDBSCAN\n(Mean |φᵢⱼ| dans le cluster vs global)",
                 fontsize=13, fontweight='bold')

    global_mean = np.abs(shap2).mean(0)

    for ax, cluster_id in zip(axes, unique_clusters):
        mask    = labels == cluster_id
        c_mean  = np.abs(shap2[mask]).mean(0)
        disc    = c_mean - global_mean

        # Top 6 paires discriminantes (i < j uniquement)
        pairs, scores = [], []
        for i in range(M):
            for j in range(i+1, M):
                pairs.append(f"{feature_names[i]}×{feature_names[j]}")
                scores.append(disc[i, j])

        top_idx = np.argsort(scores)[::-1][:6]
        top_pairs_names  = [pairs[k]  for k in top_idx]
        top_pairs_scores = [scores[k] for k in top_idx]

        bar_colors = [GREEN if s >= 0 else RED for s in top_pairs_scores]
        bars = ax.barh(top_pairs_names[::-1], top_pairs_scores[::-1],
                       color=bar_colors[::-1], alpha=0.85, edgecolor='white')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f"Cluster {cluster_id}\n({mask.sum()} points)",
                     fontweight='bold',
                     color=COLORS[cluster_id % len(COLORS)])
        ax.set_xlabel("Discriminance\n(cluster - global)")

    #plt.tight_layout()
    plt.savefig("figures/fig6_clusters_interactions.png", bbox_inches='tight')
    plt.show()
    print("    ✓ figures/fig6_clusters_interactions.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — PCA des interactions + clustering
# ═══════════════════════════════════════════════════════════════════════════
print("  → Figure 7 : PCA + clustering...")

Zp2 = PCA(n_components=2).fit_transform(Z)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Espace des interactions ordre 2 (PCA 2D)", fontsize=13, fontweight='bold')

# Coloré par cluster HDBSCAN
ax = axes[0]
noise = labels == -1
for c in unique_clusters:
    mask = labels == c
    ax.scatter(Zp2[mask, 0], Zp2[mask, 1],
               c=COLORS[c % len(COLORS)], label=f"Cluster {c}",
               alpha=0.7, s=25, edgecolors='none')
if noise.any():
    ax.scatter(Zp2[noise, 0], Zp2[noise, 1],
               c='lightgray', label="Bruit", alpha=0.1, s=15, edgecolors='none')
ax.set_title("Clusters HDBSCAN")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=8)

# Coloré par label y
ax = axes[1]
for c, col in [(0, BLUE), (1, ORANGE)]:
    mask = y == c
    ax.scatter(Zp2[mask, 0], Zp2[mask, 1],
               c=col, label=f"y={c}", alpha=0.55, s=20, edgecolors='none')
ax.set_title("Vraies classes (y)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=9)

# Coloré par importance Shapley ordre 1 de la feature la plus importante
ax = axes[2]
top_feat = int(np.argmax(np.abs(shap1).mean(0)))
colors_s1 = shap1[:, top_feat]
sc = ax.scatter(Zp2[:, 0], Zp2[:, 1],
                c=colors_s1, cmap='RdBu_r', alpha=0.65, s=22, edgecolors='none')
plt.colorbar(sc, ax=ax, label=f"φ_{feature_names[top_feat]} (Shapley)")
ax.set_title(f"Valeur Shapley de {feature_names[top_feat]}\n(feature la + importante)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

plt.tight_layout()
plt.savefig("figures/fig7_pca_clustering.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig7_pca_clustering.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Résumé synthétique : tableau de bord
# ═══════════════════════════════════════════════════════════════════════════
print("  → Figure 8 : Tableau de bord synthétique...")

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
fig.suptitle("Tableau de bord — Comparaison des méthodes d'explicabilité",
             fontsize=14, fontweight='bold', y=1.01)

# ── [0,0] Ranking unifié ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
rankings = {
    "Shapley": np.argsort(np.abs(s1).mean(0))[::-1],
    "Owen":    np.argsort(np.abs(ow).mean(0))[::-1],
    "Winter":  np.argsort(np.abs(wi).mean(0))[::-1],
}
colors_r = [BLUE, ORANGE, PURPLE]
for idx, (name, ranking) in enumerate(rankings.items()):
    for rank, feat in enumerate(ranking):
        ax.scatter(idx, rank, c=COLORS[feat % len(COLORS)], s=120,
                   zorder=3, edgecolors='white', linewidths=0.5)
        ax.text(idx, rank, feature_names[feat], fontsize=7,
                ha='left', va='center', color='gray')
ax.set_xticks(range(3))
ax.set_xticklabels(list(rankings.keys()), fontweight='bold')
ax.set_yticks(range(M))
ax.set_yticklabels([f"Rang {i+1}" for i in range(M)], fontsize=8)
ax.set_title("Rankings d'importance\n(chaque couleur = une feature)")
ax.invert_yaxis()
ax.set_xlim(-0.5, 2.5)

# ── [0,1] Efficience ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
baseline = model.predict(background).mean()
preds    = model.predict(X[:N_COMMON])

effs = {}
for name, vals in [("Shapley", s1), ("Owen", ow), ("Winter", wi)]:
    # Vérifier que vals a la même longueur que preds
    err = np.abs(vals.sum(axis=1) - (preds - baseline)).mean()
    effs[name] = err

bars = ax.bar(list(effs.keys()), list(effs.values()),
              color=[BLUE, ORANGE, PURPLE], alpha=0.85, edgecolor='white')
for bar, val in zip(bars, effs.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title("Erreur d'efficience\n(doit être ≈ 0)")
ax.set_ylabel("|Σφ - (f(x) - E[f])|")
ax.axhline(0.01, color='red', ls='--', lw=1, label="Seuil 0.01")
ax.legend(fontsize=8)

# ── [0,2] Variance expliquée PCA ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
pca_full = PCA(n_components=min(M*(M-1)//2, N_COMMON-1))
pca_full.fit(Z)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', color=BLUE, lw=2, ms=5)
ax.axhline(80, color='orange', ls='--', lw=1.5, label="80%")
ax.axhline(95, color='red',    ls='--', lw=1.5, label="95%")
n_80 = np.searchsorted(cumvar, 80) + 1
n_95 = np.searchsorted(cumvar, 95) + 1
ax.axvline(n_80, color='orange', ls=':', lw=1)
ax.axvline(n_95, color='red',    ls=':', lw=1)
ax.set_xlabel("Nombre de composantes")
ax.set_ylabel("Variance expliquée (%)")
ax.set_title(f"PCA sur interactions ordre 2\n(80% → {n_80} PC, 95% → {n_95} PC)")
ax.legend(fontsize=8)

# ── [0,3] Taille des clusters ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
cluster_sizes = [(f"Cluster {c}", (labels==c).sum()) for c in unique_clusters]
cluster_sizes.append(("Bruit", (labels==-1).sum()))
names_c, sizes_c = zip(*cluster_sizes)
bar_colors_c = [COLORS[i % len(COLORS)] for i in range(len(unique_clusters))] + ['lightgray']
ax.bar(names_c, sizes_c, color=bar_colors_c, alpha=0.85, edgecolor='white')
ax.set_ylabel("Nombre d'observations")
ax.set_title("Taille des clusters HDBSCAN\n(sur interactions ordre 2)")
ax.tick_params(axis='x', rotation=20)
for i, v in enumerate(sizes_c):
    ax.text(i, v + 1, str(v), ha='center', fontsize=9, fontweight='bold')

# ── [1,:] Contributions Owen par groupe ──────────────────────────────────
ax = fig.add_subplot(gs[1, :2])
group_labels = [f"Groupe {k}\n({[feature_names[f] for f in g]})"
                for k, g in enumerate(fine_groups)]
contrib_per_group_s1 = [np.abs(s1[:, g]).mean() for g in fine_groups]
contrib_per_group_ow = [np.abs(ow[:, g]).mean() for g in fine_groups]
contrib_per_group_wi = [np.abs(wi[:, g]).mean() for g in fine_groups]

x_pos = np.arange(len(fine_groups))
w = 0.25
ax.bar(x_pos - w, contrib_per_group_s1, w, label="Shapley", color=BLUE,   alpha=0.85)
ax.bar(x_pos,     contrib_per_group_ow, w, label="Owen",    color=ORANGE, alpha=0.85)
ax.bar(x_pos + w, contrib_per_group_wi, w, label="Winter",  color=PURPLE, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(group_labels, fontsize=8)
ax.set_ylabel("Mean |valeur| du groupe")
ax.set_title("Contribution par groupe\nShapley vs Owen vs Winter")
ax.legend(fontsize=9)

# ── [1,2:] Divergence par feature ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2:])
div_ow = np.abs(ow - s1).mean(0)
div_wi = np.abs(wi - s1).mean(0)
x_pos2 = np.arange(M)
ax.bar(x_pos2 - 0.2, div_ow, 0.35, label="Owen - Shapley",   color=ORANGE, alpha=0.85)
ax.bar(x_pos2 + 0.2, div_wi, 0.35, label="Winter - Shapley", color=PURPLE, alpha=0.85)
ax.set_xticks(x_pos2)
ax.set_xticklabels(feature_names, rotation=45, fontsize=9)
ax.set_ylabel("Mean |divergence|")
ax.set_title("Divergence vs Shapley par feature\n(fort = effet de groupe significatif)")
ax.legend(fontsize=9)
ax.axhline(div_ow.mean(), color=ORANGE, ls='--', lw=1, alpha=0.6)
ax.axhline(div_wi.mean(), color=PURPLE, ls='--', lw=1, alpha=0.6)

plt.savefig("figures/fig8_dashboard.png", bbox_inches='tight')
plt.show()
print("    ✓ figures/fig8_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ TEXTE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RÉSUMÉ DES RÉSULTATS")
print("=" * 60)

tau_so, _ = kendalltau(np.abs(s1).mean(0), np.abs(ow).mean(0))
tau_sw, _ = kendalltau(np.abs(s1).mean(0), np.abs(wi).mean(0))
tau_ow, _ = kendalltau(np.abs(ow).mean(0), np.abs(wi).mean(0))

top_feat_shap = feature_names[np.argmax(np.abs(s1).mean(0))]
top_pair_idx  = np.unravel_index(np.argmax(np.abs(shap2).mean(0)), (M, M))
top_pair      = f"{feature_names[top_pair_idx[0]]} × {feature_names[top_pair_idx[1]]}"
top_div_feat  = feature_names[np.argmax(div_ow)]

print(f"\n  Feature la + importante (Shapley)  : {top_feat_shap}")
print(f"  Paire la + interactive (ordre 2)   : {top_pair}")
print(f"  Feature à + forte divergence Owen  : {top_div_feat}")
print(f"\n  Kendall τ Shapley vs Owen          : {tau_so:.3f}")
print(f"  Kendall τ Shapley vs Winter        : {tau_sw:.3f}")
print(f"  Kendall τ Owen vs Winter           : {tau_ow:.3f}")
print(f"\n  Clusters HDBSCAN                   : {n_clusters}")
print(f"  Points de bruit                    : {(labels==-1).sum()}")

print("\n  Figures générées :")
for i in range(1, 9):
    path = f"figures/fig{i}_*.png"
    import glob
    files = glob.glob(f"figures/fig{i}_*.png")
    if files:
        print(f"    ✓ {files[0]}")

print("\nTerminé.")