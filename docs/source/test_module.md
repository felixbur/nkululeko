# Using Split Train and Test Data

This tutorial shows how to configure separate train and test sets using the `split_strategy` option, and how to confirm results using the test module.

**Reference**: [How to use train, dev and test splits with Nkululeko](http://blog.syntheticspeech.de/2024/04/26/how-to-use-train-dev-and-test-splits-with-nkululeko/)

## Overview

In machine learning, the typical workflow is:

1. **Train** your model on a training set
2. **Tune** hyperparameters on a development (validation) set  
3. **Evaluate** on a held-out test set

Nkululeko can directly handle train/test splits in a single experiment using the `split_strategy` option.

## Using Split Strategy

The simplest way to define train and test sets is using separate databases with `split_strategy`:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_splits
save = True

[DATA]
databases = ['train', 'dev', 'test']
train = ./data/polish/polish_train.csv
train.type = csv
train.absolute_path = False
train.split_strategy = train
dev = ./data/polish/polish_dev.csv
dev.type = csv
dev.absolute_path = False
dev.split_strategy = train
test = ./data/polish/polish_test.csv
test.type = csv
test.absolute_path = False
test.split_strategy = test
target = emotion

[FEATS]
type = ['os']
scale = standard

[MODEL]
type = xgb
save = True
```

### Key Configuration

| Option | Description |
|--------|-------------|
| `<db>.split_strategy = train` | Use this database for training |
| `<db>.split_strategy = test` | Use this database for testing only |
| `<db>.split_strategy = dev` | Use this database for development/validation |

## Running the Experiment

With split strategies defined, run a single command:

```bash
python -m nkululeko.nkululeko --config myconf.ini
```

This trains on the train/dev data and evaluates on the test set in one go.

## Confirming Results with the Test Module

After running your experiment, you can use the **test module** to re-evaluate your saved model on the test set. This is useful to:

- Confirm the results from your experiment
- Evaluate the model on additional test sets
- Generate detailed test reports

### Using the Test Module

First, ensure your model is saved by setting `save = True` in both `[EXP]` and `[MODEL]` sections. Then run:

```bash
python -m nkululeko.test --config myconf.ini
```

The test module will:
1. Load the saved best model
2. Evaluate it on the test set(s)
3. Generate results with `test` suffix in the output files

### Defining Test Databases

You can also specify test databases using the `tests` option:

```ini
[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = speaker_split
target = emotion
labels = ['anger', 'happiness', 'neutral', 'sadness']
; Define additional test databases
tests = ['crema-d']
crema-d = ./data/crema-d/crema-d
crema-d.split_strategy = test
```

## Cross-Database Evaluation

A common use case is training on one database and testing on another:

```ini
[DATA]
databases = ['emodb', 'crema-d']
emodb = ./data/emodb/emodb
emodb.split_strategy = train
target = emotion
labels = ['anger', 'happiness']
; Test on a different database
crema-d = ./data/crema-d/crema-d
crema-d.split_strategy = test
```

This evaluates how well your model generalizes to unseen data from a different source.

## Example Files

- [`exp_emodb_os_xgb_test.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_os_xgb_test.ini): Cross-database evaluation example
- [`exp_polish_flags.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_polish_flags.ini): Flags module for systematic comparison

## Tips

1. **Save your model**: Set `save = True` in both `[EXP]` and `[MODEL]` sections to use the test module
2. **Use split_strategy consistently**: Set `train`, `dev`, or `test` for each database
3. **Matching labels**: Ensure test database has the same labels as training data
4. **Final evaluation**: Only evaluate on test set once, after all hyperparameter tuning is complete

## Comparing Multiple Configurations with Flags Module

To systematically compare different models, features, and preprocessing options, use the **flags module**:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_flags

[DATA]
databases = ['train', 'dev', 'test']
train = ./data/polish/polish_train.csv
train.type = csv
train.split_strategy = train
dev = ./data/polish/polish_dev.csv
dev.type = csv
dev.split_strategy = train
test = ./data/polish/polish_test.csv
test.type = csv
test.split_strategy = test
target = emotion

[FEATS]
; Leave empty - will be set by FLAGS

[MODEL]
; Leave empty - will be set by FLAGS

[FLAGS]
models = ['xgb', 'svm']  
features = ['praat', 'os']   
balancing = ['none', 'ros', 'smote']  
scale = ['none', 'standard', 'robust', 'minmax']  
```

### Running the Flags Module

```bash
python -m nkululeko.flags --config exp_polish_flags.ini
```

This will automatically run all combinations:
- 2 models × 2 feature sets × 3 balancing methods × 4 scalers = 48 experiments

### Flags Options

| Flag | Values | Description |
|------|--------|-------------|
| `models` | `['svm', 'xgb', 'mlp', 'knn', ...]` | Model types to compare |
| `features` | `['os', 'praat', 'wav2vec2', ...]` | Feature extractors |
| `balancing` | `['none', 'ros', 'smote', ...]` | Class balancing methods |
| `scale` | `['none', 'standard', 'robust', 'minmax']` | Feature scaling |

## Related Tutorials

- [Train/Dev/Test Splits](traindevtest.md): Automatic three-way splitting with `traindevtest = True`
- [Comparing Runs](compare_runs.md): Statistical comparison of multiple experiments
