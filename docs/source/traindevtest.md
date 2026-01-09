# Train/Dev/Test Splits

This tutorial explains how to use three-way data splits (train, dev, test) in Nkululeko for proper model evaluation and to avoid overfitting.

**Reference**: [Nkululeko: How to use train/dev/test splits](http://blog.syntheticspeech.de/2025/03/31/nkululeko-how-to-use-train-dev-test-splits/)

## Why Three Splits?

Supervised machine learning works as follows:

1. **Training phase**: A learning algorithm adapts to a training dataset, producing a trained model
2. **Inference phase**: The model makes predictions on a test set

### The Overfitting Problem

Complex models may **memorize** training data rather than learning generalizable patterns. This means:
- ✅ Great performance on training data
- ❌ Poor performance on new data

This phenomenon is called **overfitting**.

### The Solution: Development Set

To prevent overfitting:
- Hyperparameters are optimized using a **held-out evaluation set** (not used during training)
- Training stops when performance on the evaluation set declines (**early stopping**)
- The best-performing model on evaluation data is selected

However, this introduces a new problem: the model may now be overfitted to the evaluation data!

### The Final Solution: Test Set

A **third dataset** is needed for final testing—one that has not been used at any stage of model development.

The three splits are:
- **Train**: Used for model training
- **Dev** (Development): Used for hyperparameter tuning and early stopping
- **Test**: Used only for final evaluation

## Enabling Three-Way Splits

Enable train/dev/test splitting with a single option:

```ini
[EXP]
traindevtest = True
```

## Example: EmoDB with MLP

Here's a complete example using the emoDB dataset (which has no predefined splits):

```ini
[EXP]
root = ./examples/results/
name = exp_emodb_traindevtest
traindevtest = True
epochs = 100

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = speaker_split
labels = ['anger', 'happiness', 'neutral', 'sadness']
target = emotion

[FEATS]
type = ['os']
scale = standard

[MODEL]
type = mlp
layers = {'l1': 100, 'l2': 16}
patience = 10

[PLOT]
best_model = True
epoch_progression = True
```

### Key Options

- `traindevtest = True`: Enables three-way splitting
- `emodb.split_strategy = speaker_split`: Splits by speaker to avoid data leakage
- `patience = 10`: Early stopping patience (stops if no improvement for 10 epochs)
- `epoch_progression = True`: Plots training progress over epochs

## Split Strategies

When using `traindevtest = True`, you can use different split strategies:

### Automatic Speaker Split

```ini
emodb.split_strategy = speaker_split
```

Automatically divides speakers into train/dev/test sets.

### Manual Speaker Assignment

```ini
emodb.split_strategy = speakers_stated
emodb.train = [3, 9, 10, 11, 13, 16]
emodb.dev = [14, 8]
; Test gets remaining speakers
```

### Pre-defined Splits

For datasets with existing splits (like MELD):

```ini
[DATA]
databases = ['train', 'dev', 'test']
train = ./data/meld/meld_train.csv
train.split_strategy = train
dev = ./data/meld/meld_dev.csv
dev.split_strategy = train
test = ./data/meld/meld_test.csv
test.split_strategy = test
```

## Output and Evaluation

With `traindevtest = True`, Nkululeko produces three evaluations:

1. **Best model on dev set**: Model selected by early stopping
2. **Best model on test set**: Same model, evaluated on held-out test data
3. **Last model on dev set**: Final epoch model performance

### Interpreting Results

The test set performance is typically lower than dev set performance because:
- The model was optimized for the dev set
- The test set represents truly unseen data
- This is the most realistic estimate of real-world performance

## Example Files

- [`exp_emodb_traindevtest.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_traindevtest.ini): Basic train/dev/test with XGB
- [`exp_emodb_traindevtest_split.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_traindevtest_split.ini): Manual speaker assignment with MLP

## Running the Experiment

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_traindevtest.ini
```

## Tips

1. **Always use speaker splits**: Avoid having the same speaker in train and test sets
2. **Set patience appropriately**: Too low may stop training too early; too high wastes computation
3. **Report test set results**: Only the test set gives unbiased performance estimates
4. **Use with neural networks**: Train/dev/test splits are most important for models that can overfit (MLP, CNN, Transformers)

## Related Tutorials

- [Comparing Runs](compare_runs.md): Statistical comparison across experiments
- [Hello World](hello_world_aud.md): Basic experiment setup
