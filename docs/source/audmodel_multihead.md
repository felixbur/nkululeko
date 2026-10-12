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

`audmodel.embeddings_name` also still accepts a single bare name (e.g.
`valence`) for the original single-head behavior. As a list, all configured
heads are extracted together via one ONNX forward pass (`audonnx`'s native
`outputs=[...]`/`concat=True`), not one experiment per head. The resulting
feature columns are each head's own column(s), prefixed with the head name:
`arousal`, `valence` (one-dimensional heads keep just their own name), and
`emotion.acoustic_angry`, `emotion.acoustic_happy`, ... for a multi-dimension
head.

### 2. Add a prediction target using the same heads

```ini
[PREDICT]
targets = ['audmodel']
```

The `audmodel` PREDICT target reuses the same `FEATS.audmodel.id` /
`FEATS.audmodel.embeddings_name` keys configured above -- no separate
`[PREDICT]` key needed. For each configured head it writes one `<head>_pred`
column: the raw value for a one-dimensional head (`valence_pred`), or the
argmax class label for a multi-dimension head (`emotion.acoustic_pred`). A
row whose extraction failed is marked `unknown` for a categorical head,
rather than silently picking an arbitrary class.

### 3. Analyze the new columns

```ini
[EXPL]
value_counts = [['emotion.acoustic_pred']]
feature_distributions = True
```

`value_counts` cross-tabulates the predicted class label against the
target. `feature_distributions` runs Kruskal-Wallis/pairwise significance
tests on every extracted feature (including each head's raw value) grouped
by the target, and -- when speaker ids are available -- also produces a
speaker-level version of the same plot (one row per speaker, each feature
averaged by mean), to avoid significance being dominated by segment count
rather than real effect size.

## Full example

```ini
[EXP]
root = ./results/
name = exp_audmodel_multihead

[DATA]
databases = ['mydb']
mydb = ./data/mydb
target = depression
labels = ['depressed', 'control']

[FEATS]
type = ['audmodel']
audmodel.id = 1c250fd7-1.1.2
audmodel.embeddings_name = ['arousal', 'valence', 'emotion.acoustic']

[PREDICT]
targets = ['audmodel']

[EXPL]
sample_selection = all
value_counts = [['emotion.acoustic_pred']]
feature_distributions = True
```

Run with:

```bash
python -m nkululeko.nkululeko --config my_config.ini
```

## Output

- `store/`: extracted feature columns `arousal`, `valence`,
  `emotion.acoustic_angry`, `emotion.acoustic_happy`, ... (and so on for
  every configured head).
- Data columns (after the `audmodel` PREDICT target runs): `arousal_pred`,
  `valence_pred`, `emotion.acoustic_pred`.
- `results/<exp>/images/`: `feat_dist_*` plots per feature, plus a
  `*_speaker` variant per feature when speaker ids are available.

## Technical details

- Multi-head extraction is only supported for ONNX-backed audmodels (the
  common case). A model that falls back to the torch path (a nonzero
  `audmodel.wav2vec2.layer`, or no `model.onnx` present) has no concept of
  named output heads, and raises a clear config error if a list is given.
- A per-file extraction failure no longer aborts the whole run: it's kept
  as a missing (NaN) row, so the rest of the dataset is still usable, and
  the `audmodel` PREDICT target marks a fully-failed categorical row as
  `unknown` instead of picking an arbitrary class.
- If `onnxruntime` silently falls back to CPU despite `MODEL.device = cuda`
  (e.g. a missing cuDNN 9), a warning is logged pointing at the likely
  `LD_LIBRARY_PATH` fix.
- A typo'd or unknown head name in `audmodel.embeddings_name` raises a clear
  config error listing the model's actual valid output names, instead of a
  bare lookup error.

## Related

[`ini_file.md`](ini_file.md#feats) (`FEATS.audmodel.*` key reference),
[`explore.md`](explore.md) (`feature_distributions`/`value_counts`
mechanics), [`emotion_prediction.md`](emotion_prediction.md) (another
PREDICT target, via the standalone `nkululeko.predict` CLI).
