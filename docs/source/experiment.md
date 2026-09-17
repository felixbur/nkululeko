# nkululeko.train

The experiment module (`nkululeko.train`) orchestrates the end-to-end lifecycle of an experiment: reading an INI configuration, preparing data splits, extracting features, training models, evaluating, and producing plots / reports. This is the
the main user interface to run experiments with Nkululeko.

## Responsibilities
* Parse configuration (`[EXP]`, `[DATA]`, `[FEATS]`, `[MODEL]`, `[EXPL]`, `[PLOT]`).
* Manage run directories and result caching.
* Trigger feature extraction pipelines (opensmile, praat, wav2vec2, etc.).
* Initialize and train selected model type (svm, xgb, mlp, cnn, tree, knn, regressor variants).
* Compute metrics (accuracy, UAR, regression scores) and generate confusion matrices.
* Coordinate optional explainability steps (feature importance, distributions, regplot, PCA/t-SNE/UMAP scatter).

## Invocation
Usage:

```bash
python -m nkululeko.train --config config_file.ini
```

The config path can also be given positionally instead of via `--config`;
if neither is given, it defaults to `exp.ini` in the current directory.

Example:  

```bash
python -m nkululeko.train --config examples/exp_emodb_os_svm.ini
# equivalent
python -m nkululeko.train examples/exp_emodb_os_svm.ini
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

## Testing a New Database with an Existing Model

When `DATA.tests` is set in the config **and** a saved experiment `.pkl`
already exists, `nkululeko.train` skips training automatically and
evaluates the stored best model on the new test database instead.  This
produces a confusion matrix, a per-class text report, and a predictions CSV
with all original test columns plus a `predicted` column.

See [test_new_database.md](test_new_database.md) for a step-by-step guide.

## Combining Predictions Per Speaker (or Any Other Group)

Per-sample predictions are noisy at the individual-sample level; often the
question that actually matters is the combined judgement per speaker (or
per session, recording location, etc.) -- e.g. "majority vote across all of
this speaker's samples". Set `PLOT.combine_per_speaker` to `mode` (majority
vote) or `mean` (average, then rounded/binned into a class) to get an
extra confusion matrix plot for this combined-level view, alongside the
usual per-sample one.

```ini
[PLOT]
combine_per_speaker = mode
combine_per_speaker.col = session
```

By default the samples are grouped by the `speaker` column. Set
`combine_per_speaker.col` to group by any other column instead (e.g.
`session`), as long as it's present in the test dataframe (loaded via
`DATA.<db>.columns` if it isn't one of nkululeko's standard columns).

This produces, in the results folder:

* a confusion-matrix plot named `<exp_name>_<col>combined_<function>.png`, and
* a text file, `<col>_combined_<function>_<model_description>.txt`, with the
  combined-level score (not just the per-sample one) -- e.g.
  `speaker-combined (mode) result: UAR: 0.812 (0.750/0.870)`.

Not available for `loso`, `logo`, or k-fold cross-validation (`k_fold_cross`),
since there's no single held-out test set to combine.

## Related
See `explore.md` for dataset analysis without training and `optim.md` for hyperparameter search.
