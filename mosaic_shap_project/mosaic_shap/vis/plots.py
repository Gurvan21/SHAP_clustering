import numpy as np
import matplotlib.pyplot as plt

def scatter_2d(Z2d: np.ndarray, color: np.ndarray, title: str, path: str | None = None):
    """Simple 2D scatter (UMAP/PCA)."""
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z2d[:,0], Z2d[:,1], c=color, s=10)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=180)
    plt.show()

def heatmap_clusters_vs_interactions(Z: np.ndarray, labels: np.ndarray, interaction_names, topk: int = 20, title: str = "", path: str | None = None):
    """Heatmap: clusters (rows) vs interactions (cols).
    - Selects topk interactions globally by mean absolute value.
    - Cell = mean signed value in cluster.
    """
    labels = np.asarray(labels)
    clusters = [c for c in sorted(set(labels)) if c != -1]
    if len(clusters) == 0:
        raise ValueError("No clusters found (all points labeled -1). Try smaller min_samples/min_cluster_size.")

    mean_abs_global = np.mean(np.abs(Z[labels != -1]), axis=0)
    top = np.argsort(-mean_abs_global)[:topk]

    M = np.zeros((len(clusters), len(top)), dtype=float)
    for r, c in enumerate(clusters):
        idx = labels == c
        M[r] = np.mean(Z[idx][:, top], axis=0)

    xlabels = [interaction_names[t] for t in top]

    plt.figure(figsize=(max(8, 0.35*len(top)), max(3, 0.35*len(clusters))))
    im = plt.imshow(M, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.yticks(range(len(clusters)), [str(c) for c in clusters])
    plt.xticks(range(len(top)), xlabels, rotation=90)
    plt.title(title if title else "Clusters vs top interactions (mean signed)")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200)
    plt.show()
