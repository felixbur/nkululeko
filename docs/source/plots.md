# Nkululeko Plots

Internal plotting utilities used by `experiment` and `explore` to generate visual outputs.

## Provided Plot Types
* Confusion matrix (classification & binned regression).
* Epoch progression curves.
* Feature importance bar plots.
* Feature distributions per class (with statistical annotations).
* Dimensionality reduction scatter (PCA, t-SNE, UMAP).
* Regplot / bubble plots (see `regplot.md`).
* Bias correlation plots.
* Uncertainty (entropy) distributions.

## Key Functions (Conceptual)
| Function | Purpose |
| -------- | ------- |
| `plot_confusion_matrix` | Visual performance overview. |
| `plot_feature_importance` | Rank top contributing features. |
| `plot_feature_distribution` | Per-class density / box plots + tests. |
| `scatter_reduced` | Embedding visualization (PCA / t-SNE / UMAP). |
| `regplot` | Feature correlation (categorical/continuous target). |
| `plot_uncertainty` | Entropy-based decision confidence view. |

## Configuration Triggers
Enabled through `[EXPL]` section flags:
```ini
[EXPL]
feature_distributions = all
scatter = ['pca','tsne']
regplot = [['duration','meanF0Hz']]
uncertainty = True
```

## Output
All figures saved under `results/<exp>/images/` with consistent naming (`conf_mat.png`, `feat_importance.png`, `regplot_*`, etc.).

## Tips
* Limit plot generation for large feature sets to reduce runtime.
* Use standardized scaling to improve visual comparability.

## Related
`explore.md`, `experiment.md`, `regplot.md`.
