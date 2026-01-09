# How to Split Your Data

This tutorial explains different data splitting strategies in Nkululeko for supervised machine learning experiments. Based on the [blog post by Felix Burkhardt](http://blog.syntheticspeech.de/2022/12/01/how-to-split-you-data/).

## Why Split Data?

In supervised machine learning, you typically need three kinds of datasets:

1. **Train data**: To teach the model the relation between data and labels
2. **Dev data** (development): To tune meta-parameters of your model (e.g., number of neurons, batch size, learning rate)
3. **Test data**: To evaluate your model ONCE at the end to check on generalization

All of this is to prevent **overfitting** on your train and/or dev data. If you've used your test data for a while, you might need to find a new set, as chances are high that you overfitted on your test during experiments.

## Rules for Good Data Splits

- Train and dev can be from the same set, but the **test set is ideally from a different database**
- If you don't have much data: use an **80/20/20% split**
- If you have masses of data: use only so much dev and test that your population seems covered
- If you have really little data: use **k-fold cross-validation** for train and dev, but the test set should still be separate

## Split Strategies in Nkululeko

Nkululeko offers several split strategies configured via `split_strategy` in the `[DATA]` section:

### 1. Specified Split

Use predefined train and test files. Ideal when you have a standard benchmark dataset with official splits.

**Configuration:**
```ini
[DATA]
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
```

**Example:** [exp_emodb_split_specified.ini](../examples/exp_emodb_split_specified.ini)

**Run:**
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_specified.ini
```

**When to use:**
- You have official benchmark splits
- You want reproducible comparisons with other research
- Dataset provides predefined train/test files

---

### 2. Random Split

Randomly assign samples to train and test sets. Simple but doesn't guarantee speaker independence.

**Configuration:**
```ini
[DATA]
emodb.split_strategy = random
emodb.test_size = 20  # 20% for test
```

**Example:** [exp_emodb_split_random.ini](../examples/exp_emodb_split_random.ini)

**Run:**
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_random.ini
```

**When to use:**
- Quick experiments
- Large datasets where speaker overlap is less critical
- When speaker information is not available

**Caution:** May lead to speaker overlap between train and test, resulting in optimistic performance estimates.

---

### 3. Speaker Split

Ensures speakers in train and test are different (speaker-independent evaluation). Critical for real-world generalization.

**Configuration:**
```ini
[DATA]
emodb.split_strategy = speaker_split
emodb.test_size = 20  # 20% for test
```

**Example:** [exp_emodb_split_speaker.ini](../examples/exp_emodb_split_speaker.ini)

**Run:**
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_speaker.ini
```

**When to use:**
- Real-world deployment scenarios
- You want to test generalization to unseen speakers
- Gold standard for speaker-independent evaluation

**Why it matters:** Prevents the model from memorizing speaker characteristics, forcing it to learn genuine emotion patterns.

---

### 4. LOSO (Leave-One-Speaker-Out)

Cross-validation where each speaker is held out once as the test set. Tests generalization to every speaker.

**Configuration:**
```ini
[DATA]
emodb.split_strategy = speaker_split
emodb.test_size = 10  # Percentage for initial split

[MODEL]
logo = 10  # Number of speakers for LOSO cross-validation
```

**Example:** [exp_emodb_split_loso.ini](../examples/exp_emodb_split_loso.ini)

**Run:**
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_loso.ini
```

**When to use:**
- Small datasets with few speakers
- You want robust speaker-independent evaluation
- You need per-speaker performance analysis

**How it works:**
- Uses `speaker_split` strategy to ensure speaker independence
- The `logo` parameter specifies the number of speakers (folds)
- For EmoDB with 10 speakers, `logo = 10` means each fold leaves one speaker out (LOSO)
- Trains 10 models, each testing on a different speaker

**Note:** Computationally expensive for datasets with many speakers. The number specified in `logo` should match the number of speakers in your dataset.

---

### 5. LOGO (Leave-One-Group-Out)

Cross-validation on the training data by leaving out one group at a time. Used for meta-parameter tuning.

**Configuration:**
```ini
[DATA]
emodb.split_strategy = random  # First split train/test
emodb.test_size = 20

[MODEL]
logo = 4  # Leave-One-Group-Out with 4 groups
```

**Example:** [exp_emodb_split_logo.ini](../examples/exp_emodb_split_logo.ini)

**Run:**
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_logo.ini
```

**When to use:**
- Tuning model hyperparameters
- You want more robust validation than single train/dev split
- Combined with another split strategy for train/test

---

### 6. K-Fold Cross-Validation

Splits training data into K folds and trains K models, using each fold as validation once.

**Configuration:**
```ini
[DATA]
emodb.split_strategy = random  # First split train/test
emodb.test_size = 20

[MODEL]
k_fold_cross = 5  # 5-fold cross-validation
```

**Example:** [exp_emodb_split_kfold.ini](../examples/exp_emodb_split_kfold.ini)

**Run:**
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_kfold.ini
```

**When to use:**
- Small to medium datasets
- You want robust performance estimates
- Comparing different models or feature sets

**Common values:** 5-fold or 10-fold

---

## Exercise 1: Compare Split Strategies

Try all split methods with EmoDB using OpenSMILE features and XGBoost:

```bash
# 1. Specified split
python -m nkululeko.nkululeko --config examples/exp_emodb_split_specified.ini

# 2. Random split
python -m nkululeko.nkululeko --config examples/exp_emodb_split_random.ini

# 3. Speaker split
python -m nkululeko.nkululeko --config examples/exp_emodb_split_speaker.ini

# 4. LOSO
python -m nkululeko.nkululeko --config examples/exp_emodb_split_loso.ini

# 5. LOGO
python -m nkululeko.nkululeko --config examples/exp_emodb_split_logo.ini

# 6. 5-fold cross-validation
python -m nkululeko.nkululeko --config examples/exp_emodb_split_kfold.ini
```

**Question:** Which split strategy gives the best performance? Why?

**Expected findings:**
- **Random split** typically gives the highest performance (but least realistic)
- **Speaker split / LOSO** give more conservative (realistic) performance
- **K-fold / LOGO** provide robust estimates with confidence intervals

---

## Exercise 2: Detecting Overfitting

Run a neural network experiment to visualize when overfitting starts:

**Configuration:** [exp_emodb_split_overfitting.ini](../examples/exp_emodb_split_overfitting.ini)

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_split_overfitting.ini
```

This configuration:
- Uses an MLP with layers `{l1: 1024, l2: 64}`
- Trains for 200 epochs
- Plots epoch progression and identifies the best model

**What to look for:**
1. Open the epoch progression plot in `examples/results/exp_emodb_split_overfitting/images/`
2. Find where training loss continues decreasing but validation loss starts increasing
3. That's where overfitting begins!

---

## Comparison Table

| Split Strategy | Speaker Independent | Use Case | Computational Cost | Realism |
|---------------|-------------------|----------|-------------------|---------|
| **Specified** | Depends on dataset | Benchmark comparison | Low | Varies |
| **Random** | ❌ No | Quick experiments | Low | Low |
| **Speaker Split** | ✅ Yes | Real-world deployment | Low | High |
| **LOSO** | ✅ Yes | Small datasets, per-speaker analysis | High | Very High |
| **LOGO** | Configurable | Hyperparameter tuning | Medium | Medium |
| **K-Fold** | Configurable | Robust evaluation | Medium-High | Medium |

---

## Best Practices

### For Small Datasets (< 1000 samples)
1. Use **k-fold cross-validation** (k=5 or k=10) on train+dev
2. Keep a separate test set that you evaluate ONLY ONCE
3. Consider **LOSO** if you have < 20 speakers

### For Medium Datasets (1000-10,000 samples)
1. Use **speaker split** with 80/10/10 (train/dev/test)
2. Ensure different speakers in each split
3. Use **k-fold** on training data for hyperparameter tuning

### For Large Datasets (> 10,000 samples)
1. Simple **random split** or **speaker split** works well
2. Dev and test sets can be smaller (e.g., 5% each)
3. Focus on ensuring the test set covers the population diversity

### General Tips
- ✅ **Always** keep test data separate until final evaluation
- ✅ Use **speaker split** for realistic performance estimates
- ✅ Use **cross-validation** for robust hyperparameter tuning
- ❌ **Never** tune on test data
- ❌ Don't evaluate on test data multiple times (you'll overfit!)

---

## Advanced: Balanced Splits

For imbalanced datasets, use stratified or balanced splits:

```ini
[DATA]
emodb.split_strategy = balanced
emodb.test_size = 20
# Stratify by multiple variables with weights
balance = {'emotion':2, 'age':1, 'gender':1}
age_bins = 2
size_diff_weight = 1
```

See [exp_emodb_split.ini](../examples/exp_emodb_split.ini) for a complete example.

---

## References

- Blog post: [How to Split Your Data](http://blog.syntheticspeech.de/2022/12/01/how-to-split-you-data/)
- Documentation: [Nkululeko INI file reference](https://github.com/felixbur/nkululeko/blob/main/ini_file.md#data)
- Related: [How to Evaluate Your Model](http://blog.syntheticspeech.de/2022/11/28/how-to-evaluate-your-model/)

---

## Summary

Choosing the right split strategy is crucial for reliable machine learning experiments:

- **For benchmarking**: Use specified splits
- **For real-world deployment**: Use speaker split or LOSO  
- **For quick experiments**: Use random split
- **For small datasets**: Use k-fold cross-validation
- **For hyperparameter tuning**: Use LOGO or k-fold

Remember: Your test set performance is only meaningful if it represents the real-world scenario your model will face!
