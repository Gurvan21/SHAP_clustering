# MosaicSHAP

A small, modular project to reproduce the **same workflow as your notebooks**, but as reusable modules + runnable scripts.

## What you get
### Explainability
- **Order 1**
  - TreeSHAP (fast for tree ensembles)
  - PermutationSHAP (model-agnostic)
  - KernelSHAP (model-agnostic, slow)
- **Order 2**
  - TreeSHAP interaction values (tree ensembles)
  - Model-agnostic interaction estimators (**SHAP-IQ style**)
    - Monte-Carlo (subset-based) interaction estimator
    - Regression-based interaction estimator (surrogate over coalition masks)

### Clustering pipeline (same spirit as AntakIA notebooks)
- vectorize explanations (order1 or order2)
- optional **PCA -> ~20D**
- optional **UMAP (2D) for visualization**
- clustering: **HDBSCAN** (plus KMeans/Agglomerative if you want)
- interpretation helpers (per-cluster top features / top interactions)
- plotting helpers (UMAP scatter, heatmaps)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Run scripts (reproduce notebook-like results)
### 1) SHAP order-1 clustering
```bash
python scripts/run_order1_clustering.py --n 1200 --model rf --algo tree --min_cluster_size 60
```

### 2) SHAP order-2 interaction clustering
```bash
python scripts/run_order2_clustering.py --n 1200 --model rf --algo tree --pca_dim 20 --min_cluster_size 60
```

Figures are saved into `figures/`.

## Notes
- For binary classification, many SHAP explainers output values in **log-odds (logit) space**.
- Model-agnostic order-2 estimators can be expensive; the script uses subsampling defaults.
