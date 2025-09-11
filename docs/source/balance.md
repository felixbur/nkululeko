# Data Balancing in Nkululeko

Data imbalance is a common problem in machine learning, particularly in speech emotion recognition and other audio classification tasks. When some classes have significantly more samples than others, models tend to be biased toward the majority classes, resulting in poor performance on minority classes.

Nkululeko provides a comprehensive set of balancing techniques to address this issue through the `DataBalancer` class, supporting various over-sampling, under-sampling, and combination methods.

## Overview

The balancing functionality in nkululeko allows you to:

- **Automatically detect** class imbalance in your datasets
- **Apply various balancing techniques** to improve model performance
- **Compare different balancing methods** using the flags module
- **Preserve data integrity** while addressing imbalance issues

## Quick Start

To quickly try balancing techniques, you can use the provided demo example:

```bash
# Clone the repository and navigate to it
cd nkululeko

# Run the balancing demo with SMOTE
python -m nkululeko.nkululeko --config examples/exp_balancing_demo.ini
```

This demo uses the test dataset included with nkululeko and applies SMOTE balancing to show how class distribution changes.

To use balancing in your own nkululeko experiment, simply add the `balancing` parameter to your `[FEATS]` section:

```ini
[FEATS]
type = ['os']
balancing = smote
scale = standard
```

## Supported Balancing Methods

Nkululeko supports three categories of balancing techniques:

### 1. Over-sampling Methods
These methods increase the number of minority class samples:

- **`ros`** - Random Over-Sampling: Randomly duplicates minority class samples
- **`smote`** - SMOTE: Generates synthetic samples using k-nearest neighbors
- **`adasyn`** - ADASYN: Adaptive synthetic sampling with density-based generation
- **`borderlinesmote`** - Borderline SMOTE: Focuses on borderline samples
- **`svmsmote`** - SVM SMOTE: Uses SVM to identify support vectors for synthesis

### 2. Under-sampling Methods
These methods reduce the number of majority class samples:

- **`randomundersampler`** - Random Under-Sampling: Randomly removes majority class samples
- **`clustercentroids`** - Cluster Centroids: Replaces clusters with their centroids
- **`editednearestneighbours`** - Edited Nearest Neighbours: Removes inconsistent samples
- **`tomeklinks`** - Tomek Links: Removes Tomek link pairs

### 3. Combination Methods
These methods combine over-sampling and under-sampling:

- **`smoteenn`** - SMOTE + Edited Nearest Neighbours
- **`smotetomek`** - SMOTE + Tomek Links

## Working Examples

### Basic SMOTE Balancing

```ini
[EXP]
root = ./examples/results/
name = exp_smote_balancing
runs = 1
epochs = 10

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
target = emotion
labels = ['anger', 'happiness', 'sadness', 'neutral']

[FEATS]
type = ['os']
balancing = smote
scale = standard

[MODEL]
type = xgb
```

### Random Over-Sampling Example

```ini
[EXP]
root = ./examples/results/
name = exp_ros_balancing

[DATA]
databases = ['polish_train', 'polish_dev', 'polish_test']
polish_train = ./data/polish/polish_train.csv
polish_train.type = csv
polish_train.split_strategy = train
polish_dev = ./data/polish/polish_dev.csv
polish_dev.type = csv
polish_dev.split_strategy = train
polish_test = ./data/polish/polish_test.csv
polish_test.type = csv
polish_test.split_strategy = test
target = emotion

[FEATS]
type = ['os']
balancing = ros

[MODEL]
type = svm
```

### Under-sampling with Cluster Centroids

```ini
[EXP]
root = ./examples/results/
name = exp_clustercentroids_balancing

[DATA]
databases = ['train', 'dev', 'test']
train = ./data/polish/polish_train.csv
train.type = csv
train.split_strategy = train
target = emotion

[FEATS]
type = ['praat']
balancing = clustercentroids
scale = robust

[MODEL]
type = mlp
```

## Choosing the Right Balancing Method

### When to Use Over-sampling

**Use over-sampling when:**
- You have limited data and don't want to lose samples
- The minority classes have sufficient diversity to generate meaningful synthetic samples
- Computational resources allow for larger datasets

**Recommended methods:**
- **SMOTE**: Good general-purpose choice, works well with most datasets
- **ADASYN**: Better for highly imbalanced datasets
- **ROS**: Simple and fast, good baseline

### When to Use Under-sampling

**Use under-sampling when:**
- You have abundant data in majority classes
- Computational resources are limited
- The majority class contains redundant or noisy samples

**Recommended methods:**
- **Cluster Centroids**: Preserves data distribution while reducing size
- **Tomek Links**: Removes noisy and borderline samples
- **Random Under-sampling**: Simple and fast baseline

### When to Use Combination Methods

**Use combination methods when:**
- You want the benefits of both approaches
- The dataset has complex imbalance patterns
- You need to clean noisy samples while adding synthetic ones

**Recommended methods:**
- **SMOTE + Tomek**: Generates synthetic samples then removes noise
- **SMOTE + ENN**: More aggressive noise removal

## Comparing Balancing Methods

Use the flags module to systematically compare different balancing techniques:

```ini
[EXP]
root = ./examples/results/
name = balancing_comparison

[DATA]
databases = ['mydata']
mydata = ./data/mydata.csv
target = emotion

[FEATS]
type = ['os']
scale = ['standard']

[MODEL]
type = ['xgb']

[FLAGS]
balancing = ['none', 'ros', 'smote', 'adasyn', 'clustercentroids', 'smoteenn']
```

This will run experiments with all specified balancing methods and show you which performs best on your dataset.

## Understanding the Output

When balancing is applied, you'll see output like this:

```
Balancing features with: smote
Original dataset size: 1200
Original class distribution: {'happy': 400, 'sad': 300, 'angry': 300, 'neutral': 200}
Balanced dataset size: 1600 (was 1200)
New class distribution: {'happy': 400, 'sad': 400, 'angry': 400, 'neutral': 400}
Class distribution after smote balancing: {'happy': 400, 'sad': 400, 'angry': 400, 'neutral': 400}
```

### Key Information:
- **Original size**: Number of samples before balancing
- **Original distribution**: Number of samples per class before balancing
- **Balanced size**: Number of samples after balancing
- **New distribution**: Number of samples per class after balancing

## Advanced Configuration

### Custom Random State

The balancing process uses a random state for reproducibility. You can control this in your experiment configuration:

```python
# This is handled automatically by nkululeko's experiment setup
# The random state is derived from your experiment configuration
```

### Method-Specific Parameters

Some balancing methods accept additional parameters. While nkululeko uses sensible defaults, you can modify the source code for custom behavior:

```python
# Example: Custom SMOTE configuration
sampler = SMOTE(
    random_state=self.random_state,
    k_neighbors=5,  # Number of neighbors for synthesis
    sampling_strategy='auto'  # Which classes to balance
)
```

## Best Practices

### 1. Start with SMOTE
SMOTE is a good default choice for most audio classification tasks:
```ini
[FEATS]
balancing = smote
```

### 2. Consider Data Size
- **Small datasets** (< 1000 samples): Use over-sampling (ROS, SMOTE)
- **Large datasets** (> 10000 samples): Consider under-sampling (cluster centroids)
- **Medium datasets**: Try combination methods (SMOTE + Tomek)

### 3. Validate on Separate Test Set
Always ensure your test set remains unbalanced to get realistic performance estimates:

```ini
[DATA]
# Training data will be balanced
train.split_strategy = train
# Test data remains unbalanced
test.split_strategy = test
```

### 4. Monitor Class Distribution
Check the balancing output to ensure the method is working as expected:
- Over-sampling should increase dataset size
- Under-sampling should decrease dataset size
- Check that target classes are actually balanced

### 5. Compare Multiple Methods
Use the flags module to systematically compare balancing approaches:

```ini
[FLAGS]
balancing = ['none', 'smote', 'adasyn', 'clustercentroids']
models = ['xgb']
features = ['os']
```

## Troubleshooting

### Common Issues

1. **"Unknown balancing algorithm" error**
   - Check spelling of the balancing method name
   - Ensure the method is in the supported list

2. **Memory errors with over-sampling**
   - Try under-sampling methods instead
   - Reduce feature dimensions before balancing

3. **Poor results after balancing**
   - Try different balancing methods
   - Check if your features are suitable for the chosen method
   - Ensure test set remains unbalanced

4. **SMOTE failing with sparse data**
   - Try ADASYN instead of SMOTE
   - Increase the number of samples in minority classes
   - Use ROS as a fallback

### Error Messages

```python
# If you see this error:
"Unknown balancing algorithm: invalid_method"
# Available methods: ['ros', 'smote', 'adasyn', ...]

# Solution: Use one of the supported methods
balancing = smote  # instead of 'invalid_method'
```

## Integration with Other Features

### With Feature Scaling
Balancing works well with feature scaling:

```ini
[FEATS]
type = ['os']
balancing = smote
scale = standard  # Apply scaling after balancing
```

### With Multiple Features
Balancing is applied to the combined feature space:

```ini
[FEATS]
type = ['os', 'praat']  # Features are combined first
balancing = adasyn      # Then balancing is applied
```

### With Cross-Validation
Balancing is applied within each fold to prevent data leakage:

```ini
[EXP]
runs = 5  # Each run applies balancing independently
```

## Performance Considerations

### Computational Impact
- **Over-sampling**: Increases dataset size → longer training time
- **Under-sampling**: Decreases dataset size → faster training time
- **Combination methods**: Variable impact depending on data distribution

### Memory Usage
- **SMOTE/ADASYN**: May require significant memory for large datasets
- **Cluster Centroids**: Reduces memory requirements
- **ROS**: Minimal additional memory (just duplicates existing samples)

### Timing
The balancing process typically adds minimal overhead compared to feature extraction and model training.

## Real-World Examples

### Emotion Recognition with Imbalanced Data

```ini
# Example: EmoDb dataset with balanced emotions
[EXP]
root = ./results/emotion_balanced/
name = emodb_balanced_experiment

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
target = emotion
labels = ['anger', 'happiness', 'sadness', 'neutral']

[FEATS]
type = ['os']
set = eGeMAPSv02
balancing = smote
scale = standard

[MODEL]
type = xgb
save = True

[PLOT]
name = balanced_emotion_results
```

### Age Detection with Severe Imbalance

```ini
# Example: Age groups with severe imbalance
[EXP]
name = age_detection_balanced

[DATA]
databases = ['age_data']
age_data = ./data/age/age_dataset.csv
target = age_group
labels = ['child', 'adult', 'elderly']

[FEATS]
type = ['praat']
balancing = adasyn  # ADASYN works well with severe imbalance
scale = robust

[MODEL]
type = svm
kernel = rbf
```

### Comparing All Methods

```ini
# Example: Systematic comparison of all balancing methods
[EXP]
root = ./results/balancing_study/
name = comprehensive_balancing_study

[DATA]
databases = ['study_data']
study_data = ./data/study/dataset.csv
target = label

[FEATS]
type = ['os']
scale = ['standard']

[MODEL]
type = ['xgb']

[FLAGS]
balancing = [
    'none', 'ros', 'smote', 'adasyn', 'borderlinesmote',
    'randomundersampler', 'clustercentroids', 'tomeklinks',
    'smoteenn', 'smotetomek'
]
```

## Conclusion

Data balancing is a crucial step in building robust audio classification models. Nkululeko's comprehensive balancing support allows you to:

1. **Easily apply** various balancing techniques with a single configuration parameter
2. **Systematically compare** different methods using the flags module
3. **Maintain reproducibility** with consistent random states
4. **Monitor results** with detailed logging and distribution reporting

Start with SMOTE for most applications, but don't hesitate to explore other methods based on your specific dataset characteristics and computational constraints. The flags module makes it easy to find the optimal balancing strategy for your particular use case.
