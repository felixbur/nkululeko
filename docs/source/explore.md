# nkululeko.explore

The `nkululeko.explore` module focuses on exploratory data analysis and feature examination without full model training (unless requested). Use it to inspect distributions, correlations, and dimensionality reduction plots early.

## Features
* Feature importance (tree / xgb model quick fit).
* Feature distributions per category (with statistical tests like Mann-Whitney, t-test, Levene).
* Scatter plots (PCA, t-SNE, UMAP) for feature space structure.
* Regression plots (`regplot`) between feature pairs (categorical or continuous targets).
* Bias / correlation plots between automatically predicted properties and target labels.

## Invocation
```bash
python -m nkululeko.explore --config examples/exp_emodb_explore_features.ini
```

## Minimal INI Example
```ini
[EXP]
name = results/exp_explore

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb.csv
target = emotion

[FEATS]
type = ['praat']
scale = standard

[EXPL]
feature_distributions = all
scatter = ['pca','tsne']
regplot = [['duration','meanF0Hz']]
print_stats = True
```

## Outputs
Plots saved to `results/<exp>/images/` (feat_importance, feat_dist, regplot_*, tsne.png, pca.png, etc.). Statistical summaries in log output.

## Tips
* Use `max_feats` to limit heavy plots on large feature sets.
* Enable `print_stats` for regression plot statistical output.
* Combine scatter methods (`pca`, `tsne`, `umap`) for complementary structure views.

## Related
`regplot.md` (detailed correlation plotting), `experiment.md` (full pipeline).
