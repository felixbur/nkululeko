# Testing a New Database with an Already Trained Model

Sometimes you have already trained a model on one or more databases and want to
evaluate it on a completely different database — without repeating the
(potentially expensive) training step.

The `nkululeko.nkululeko` module detects this situation automatically: if
`DATA.tests` is set in the configuration **and** a saved experiment file
already exists on disk, it skips training entirely, loads the stored best model,
and evaluates it on the new test database.

## Prerequisites

1. A completed experiment with `[EXP] save = True` so the experiment state was
   written to disk.
2. A second INI file (or the same one extended with `DATA.tests`) that points
   at the new database.

## Step 1 — Train and save the model

Make sure your original experiment config has saving enabled:

```ini
[EXP]
root = ./results/mymodel
name = exp_emodb_mlp
save = True          ; <-- required

[DATA]
databases = ['emodb']
emodb     = ./data/emodb/emodb
emodb.split_strategy = speaker_split
target = emotion
labels = ['angry', 'happy', 'sad', 'neutral']

[FEATS]
type = ['audwav2vec2']

[MODEL]
type = mlp
layers = [1024, 64]
save = True          ; <-- keep individual epoch files too
```

Run training once:

```bash
python -m nkululeko.nkululeko --config exp_emodb_mlp.ini
```

The experiment is saved to
`results/mymodel/exp_emodb_mlp/store/emodb_emotion_mlp_audwav2vec2_64-1024.pkl`.

## Step 2 — Create a test config

Create a new INI file (e.g. `test_ravdess.ini`) that **reuses the same
`[EXP]`, `[FEATS]`, and `[MODEL]` sections** (so the save path resolves to the
same file) and adds a `DATA.tests` entry for the new database:

```ini
[EXP]
root = ./results/mymodel
name = exp_emodb_mlp
save = True

[DATA]
databases = ['emodb']           ; training databases (unchanged)
emodb     = ./data/emodb/emodb
emodb.split_strategy = speaker_split
target = emotion
labels = ['angry', 'happy', 'sad', 'neutral']

; ---- new test database ----
tests  = ['ravdess']
ravdess = ./data/ravdess/ravdess_test.csv
ravdess.type = csv
ravdess.absolute_path = False
ravdess.split_strategy = test
ravdess.mapping = {'angry':'angry', 'happy':'happy', 'sad':'sad', 'neutral':'neutral'}

[FEATS]
type = ['audwav2vec2']

[MODEL]
type = mlp
layers = [1024, 64]
save = True
```

> **Label mapping**: use `<db>.mapping` to align the new database's emotion
> labels to the names used during training.  Only labels present in the
> training `labels` list will be kept.

## Step 3 — Run evaluation

```bash
python -m nkululeko.nkululeko --config test_ravdess.ini
```

Because the saved experiment file exists, nkululeko prints:

```
DEBUG: nkululeko: DATA.tests is set and saved experiment found at
       results/mymodel/exp_emodb_mlp/store/emodb_emotion_mlp_….pkl
       — loading best model, skipping training
```

No training takes place.  The run goes straight to evaluation.

## Outputs

After the run the following files are written to the results directory:

| File | Description |
|------|-------------|
| `…_ravdess_0_NNN_cnf.png` | Confusion matrix with string label names |
| `…_ravdess_0_NNN_predictions.txt` | Per-class precision / recall / UAR |
| `…_ravdess_0_NNN_predictions.csv` | All original test columns + a `predicted` column |

The CSV contains one row per audio segment with the original columns from the
test database (file path, ground-truth emotion, any speaker/gender metadata)
and an additional `predicted` column with the decoded model prediction.

The path is logged at the end of the run:

```
DEBUG: nkululeko: predictions CSV saved to: results/mymodel/exp_emodb_mlp/results/run_0/…_predictions.csv
```

## How it works

The detection logic in `nkululeko.nkululeko.doit()`:

1. Checks whether `DATA.tests` is non-empty in the config.
2. Resolves the expected save path via `util.get_save_name()`.
3. If the file exists → fast path (no training):
   - `experiment.load(save_name)` — restores the full experiment state.
   - `experiment.fill_tests(encode=False)` — loads test data keeping original
     string labels.
   - `experiment.extract_test_feats()` — extracts features for the test set.
   - `runmgr.get_best_model()` — loads the best model checkpoint from disk.
   - `model.predict()` — runs inference and computes metrics.
   - Produces confusion matrix, text report, and predictions CSV.
4. If the file does **not** exist → normal training flow (first run).

This means the **same config file** works for both training (first run) and
evaluation (subsequent runs): the first invocation trains and saves; every
subsequent invocation with `DATA.tests` set evaluates without retraining.

## Tips

- **Regenerate test features**: if you change `FEATS` settings, delete or
  rename `store/extra_testdf.csv` inside the experiment root, or set
  `DATA.no_reuse = True` to force re-extraction.
- **Multiple test databases**: list several databases in `DATA.tests` to
  evaluate on all of them in a single run:
  ```ini
  tests = ['ravdess', 'emovo', 'cremad']
  ```
  All test databases are concatenated for a single combined evaluation.
- **Force retraining**: remove the `.pkl` file from the store directory or
  rename it to make the experiment file disappear — nkululeko will then
  fall back to the normal training path.

## Related

- [experiment.md](experiment.md) — full reference for `nkululeko.nkululeko`
- [test_module.md](test_module.md) — train/dev/test split workflow
- [predict.md](predict.md) — `nkululeko.predict` for file-list based inference
- [multidb.md](multidb.md) — training across multiple databases
