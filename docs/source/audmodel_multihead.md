# Multi-head audmodel Extraction and Prediction

This tutorial demonstrates how to extract several output heads of one
[audmodel](https://audeering.github.io/audmodel/index.html) (e.g. audEERING's
devAIce expression model) in a single pass, and turn them into prediction
columns for analysis.

## Overview

A devAIce-style model can expose several named output heads from one audio
pass, e.g. `arousal`, `valence`, `emotion.acoustic` (7 classes),
`basic-4.acoustic` (4 classes). Without multi-head support, correlating a
target against several of these heads took one full experiment -- one pass
over the audio -- per head. This is useful for:

- Correlating a target (e.g. a clinical label) with several emotional
  dimensions and classification heads of the same model.
- Reusing a single audio pass across many analyses instead of one per head.
- Feeding several heads' predictions into `value_counts`/`feature_distributions`
  without writing custom scripts.

## Quick Start

### 1. Extract several heads in one pass

```ini
[FEATS]
type = ['audmodel']
audmodel.id = 1c250fd7-1.1.2
audmodel.embeddings_name = ['arousal', 'valence', 'emotion.acoustic']
```

Runnable example: [`examples/exp_emodb_audmodel_multihead.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_audmodel_multihead.ini).

`audmodel.embeddings_name` also still accepts a single bare name (e.g.
`valence`) for the original single-head behavior. As a list, all configured
heads are extracted together via one ONNX forward pass (`audonnx`'s native
`outputs=[...]`/`concat=True`), not one experiment per head. The resulting
feature columns are each head's own column(s), prefixed with the head name:
`arousal`, `valence` (one-dimensional heads keep just their own name), and
`emotion.acoustic_angry`, `emotion.acoustic_happy`, ... for a multi-dimension
head.

### 2. Add predicted columns with nkululeko.predict

```bash
python -m nkululeko.predict \
    --list ./your_dataset.csv \
    --model audmodel \
    --config your_feats_config.ini \
    --outfile ./your_dataset_with_audmodel.csv
```

`your_feats_config.ini` only needs the `[FEATS] audmodel.id`/
`audmodel.embeddings_name` keys from step 1 -- the `audmodel` autopredict
target reuses them, no separate config key needed. For each configured
head it writes one `<head>_pred` column onto the CSV: the raw value for a
one-dimensional head (`valence_pred`), or the argmax class label for a
multi-dimension head (`emotion.acoustic_pred`). A row whose extraction
failed is marked `unknown` for a categorical head, rather than silently
picking an arbitrary class. (`[PREDICT] targets = [...]` in an experiment
config, by contrast, is not wired into any runnable entry point yet --
this standalone command is the one that actually produces the columns.)

### 3. Analyze the new columns

Point a normal explore experiment at the enriched CSV from step 2 and add:

```ini
[DATA]
databases = ['mydb']
mydb = ./your_dataset_with_audmodel.csv
mydb.type = csv
target = depression

[EXPL]
value_counts = [['emotion.acoustic_pred']]
feature_distributions = True
```

`value_counts` cross-tabulates the predicted class label against the
target. `feature_distributions` runs Kruskal-Wallis/pairwise significance
tests on every extracted feature (including each head's raw value) grouped
by the target, and -- when speaker ids are available -- also produces a
speaker-level version of the same plot (each feature averaged by mean),
to avoid significance being dominated by segment count rather than real
effect size. The aggregation differs by target type: for a classification
target, one row per `(speaker, label)` -- a speaker with samples of more
than one class still contributes once per class, not collapsed into a
single speaker-wide average; for a regression target, one row per speaker
(both the label and the feature averaged), since a continuous label varies
per sample even within one speaker.

## Full example

`your_feats_config.ini` (just the keys the `audmodel` extractor/predictor
need):

```ini
[FEATS]
type = ['audmodel']
audmodel.id = 1c250fd7-1.1.2
audmodel.embeddings_name = ['arousal', 'valence', 'emotion.acoustic']
```

Step 2, add the predicted columns:

```bash
python -m nkululeko.predict \
    --list ./your_dataset.csv \
    --model audmodel \
    --config your_feats_config.ini \
    --outfile ./your_dataset_with_audmodel.csv
```

Step 3, a normal explore experiment on the enriched CSV
(`exp_audmodel_multihead.ini`):

```ini
[EXP]
root = ./results/
name = exp_audmodel_multihead
sample_selection = all

[DATA]
databases = ['mydb']
mydb = ./your_dataset_with_audmodel.csv
mydb.type = csv
target = depression
labels = ['depressed', 'control']

[FEATS]
type = ['audmodel']
audmodel.id = 1c250fd7-1.1.2
audmodel.embeddings_name = ['arousal', 'valence', 'emotion.acoustic']

[EXPL]
value_counts = [['emotion.acoustic_pred']]
feature_distributions = True
```

Run with:

```bash
python -m nkululeko.explore --config exp_audmodel_multihead.ini
```

## Output

- `your_dataset_with_audmodel.csv` (step 2): the original columns plus
  `arousal_pred`, `valence_pred`, `emotion.acoustic_pred`.
- `store/` (step 3): extracted feature columns `arousal`, `valence`,
  `emotion.acoustic_angry`, `emotion.acoustic_happy`, ... (and so on for
  every configured head).
- `results/<exp>/images/`: `feat_dist_*` plots per feature, plus a
  `*_speaker` variant per feature when speaker ids are available.

## Technical details

- Multi-head extraction is only supported for ONNX-backed audmodels (the
  common case). A model that falls back to the torch path (a nonzero
  `FEATS.wav2vec2.layer`, or no `model.onnx` present) has no concept of
  named output heads, and raises a clear config error if a list is given.
- A per-file extraction failure no longer aborts the whole run: it's kept
  as a missing (NaN) row, so the rest of the dataset is still usable, and
  the `audmodel` autopredict target marks a fully-failed categorical row
  as `unknown` instead of picking an arbitrary class (a legitimate negative
  row sum, e.g. from a model that outputs raw logits, is not mistaken for
  a failure). If the failure rate exceeds `FEATS.fail_threshold` (default
  0.5), extraction still raises a clear error instead of silently
  persisting a mostly-empty feature store.
- If `onnxruntime` silently falls back to CPU despite `MODEL.device = cuda`
  (e.g. a missing cuDNN 9), a warning is logged pointing at the likely
  `LD_LIBRARY_PATH` fix.
- A typo'd or unknown head name in `audmodel.embeddings_name` raises a clear
  config error listing the model's actual valid output names, instead of a
  bare lookup error.
- The `audmodel` autopredict target temporarily disables `FEATS.features`
  (a generic feature-subset/reorder filter) for the duration of its own
  extraction, restoring it afterward: that filter would otherwise
  reshuffle or drop audmodel's own output columns before the predictor
  can turn them into `<head>_pred` columns.

## Related

[`ini_file.md`](ini_file.md#feats) (`FEATS.audmodel.*` key reference),
[`explore.md`](explore.md) (`feature_distributions`/`value_counts`
mechanics), [`emotion_prediction.md`](emotion_prediction.md) (another
PREDICT target, via the standalone `nkululeko.predict` CLI).
