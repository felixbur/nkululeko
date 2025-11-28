# nkululeko.experiment

The `nkululeko.experiment` module orchestrates the end-to-end lifecycle of an experiment: reading an INI configuration, preparing data splits, extracting features, training models, evaluating, and producing plots / reports.

## Responsibilities
* Parse configuration (`[EXP]`, `[DATA]`, `[FEATS]`, `[MODEL]`, `[EXPL]`, `[PLOT]`).
* Manage run directories and result caching.
* Trigger feature extraction pipelines (opensmile, praat, wav2vec2, etc.).
* Initialize and train selected model type (svm, xgb, mlp, cnn, tree, knn, regressor variants).
* Compute metrics (accuracy, UAR, regression scores) and generate confusion matrices.
* Coordinate optional explainability steps (feature importance, distributions, regplot, PCA/t-SNE/UMAP scatter).

## Invocation
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_os_svm.ini
```

## Key Concepts
| Concept | Description |
| ------- | ----------- |
| Runs | Repeats with different seeds for robustness. |
| Store Format | Choice of cached feature file format (csv, feather, pickle). |
| Scaling | Feature normalization (standard, minmax, none). |
| Augmentation | Optional audio transforms before extraction. |

## Common INI Snippet
```ini
[EXP]
name = results/exp_demo
runs = 1

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb.csv
target = emotion

[FEATS]
type = ['praat']
scale = standard

[MODEL]
type = xgb

[EXPL]
conf_mat = True
feature_distributions = top
regplot = [['duration','meanF0Hz']]
```

## Outputs
* `results/<exp>/images/` – plots (confusion matrix, distributions, regplots).
* `results/<exp>/results/` – metrics summaries.
* Feature cache under experiment root.

## Internals
Important classes/functions (high-level):
* Experiment class – central coordinator.
* Hooks for plotting via `plots` module.
* Label encoding/decoding abstraction to support consistent plotting.

## Tips
1. Start with a single feature set and model to validate pipeline.
2. Enable caching to save time on subsequent runs.
3. Use balanced splits (`speaker_split`) for speaker leakage prevention.
4. Limit `max_feats` when exploring importance to keep plots readable.

## Related
See `explore.md` for dataset analysis without training and `optim.md` for hyperparameter search.
