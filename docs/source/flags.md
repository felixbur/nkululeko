# Nkululeko Flags Module Tutorial

The **flags module** in nkululeko allows you to run multiple experiments with different parameter combinations automatically. This is particularly useful for hyperparameter tuning and systematic exploration of different configurations.

## Overview

Instead of manually creating multiple configuration files and running experiments one by one, the flags module enables you to:

- Define multiple values for different parameters in a single configuration file
- Automatically generate all possible combinations of these parameters
- Run all experiments sequentially with optimized feature extraction
- Get a summary of results and identify the best performing configuration

## Quick Start Demo

To quickly try the flags module, you can use the provided demo example:

```bash
# Clone the repository and navigate to it
cd nkululeko

# Run the flags demo (this will test 2×2×2×2 = 16 combinations)
python -m nkululeko.flags --config examples/exp_flags_demo.ini
```

This demo uses the test dataset included with nkululeko and will show you how the flags module works with a small, manageable number of experiments.

## Basic Usage

### 1. Configuration File Structure

To use the flags module, you need to add a `[FLAGS]` section to your standard nkululeko configuration file. Here's the basic structure:

```ini
[EXP]
root = /tmp/results/
name = my_flags_experiment
runs = 1
epochs = 10

[DATA]
databases = ['mydata']
mydata = ./data/mydata.csv
mydata.type = csv
target = emotion

[FLAGS]
models = ['xgb', 'svm', 'mlp']
features = ['os', 'praat', 'mfcc']   
balancing = ['none', 'ros', 'smote']  
scale = ['none', 'standard', 'robust']
```

### 2. Running with Flags

There are several ways to run experiments with flags:

#### Method 1: Command line with --flags argument
```bash
python -m nkululeko.flags --config my_config.ini --flags
```

#### Method 2: Automatic detection
If your configuration file contains a `[FLAGS]` section, nkululeko will automatically detect it:
```bash
python -m nkululeko.flags --config my_config.ini
```

#### Method 3: Command line parameter override
You can also override specific parameters from the command line:
```bash
# Override model and feature types
python -m nkululeko.flags --config my_config.ini --model xgb --feat os

# Specify multiple values for testing
python -m nkululeko.flags --config my_config.ini --model xgb svm --feat "['os', 'praat']"

# Combine with other parameters
python -m nkululeko.flags --config my_config.ini --balancing smote --scale standard --epochs 50
```

### 3. Module Selection

The flags module supports different nkululeko modules:

```bash
# Standard training (default)
python -m nkululeko.flags --config my_config.ini --mod nkulu

# Testing mode
python -m nkululeko.flags --config my_config.ini --mod test
```

## Supported Flag Parameters

The flags module supports the following parameters:

### Core Parameters
- **`models`**: List of model types to test
  - Example: `['xgb', 'svm', 'mlp', 'tree']`
  
- **`features`**: List of feature types to test
  - Example: `['os', 'praat', 'mfcc', 'hubert', 'wav2vec2']`
  
- **`balancing`**: List of balancing methods to test
  - Example: `['none', 'ros', 'smote', 'adasyn']`
  
- **`scale`**: List of scaling methods to test
  - Example: `['none', 'standard', 'robust', 'minmax']`

### Custom Parameters
You can also define custom parameters using prefixes:
- **`model_*`**: Model-specific parameters (e.g., `model_learning_rate`)
- **`feats_*`**: Feature-specific parameters (e.g., `feats_set`)
- **`exp_*`**: Experiment-specific parameters (e.g., `exp_epochs`)

## Working Example

Let's look at a complete working example based on the Polish emotion dataset:

```ini
[EXP]
root = /tmp/results/
name = exp_polish_flags1

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

[FLAGS]
models = ['xgb', 'svm']
features = ['praat', 'os']   
balancing = ['none', 'ros', 'smote']  
scale = ['none', 'standard', 'robust', 'minmax']
```

This configuration will run **2 × 2 × 3 × 4 = 48 experiments** testing all combinations of:
- 2 models (XGBoost, SVM)
- 2 feature types (Praat, OpenSMILE)
- 3 balancing methods (none, ROS, SMOTE)
- 4 scaling methods (none, standard, robust, minmax)

## Advanced Usage

### 1. Single Value Parameters
If you want to test a single value, you can specify it as a string without brackets:
```ini
[FLAGS]
models = 'xgb'
features = ['os', 'praat']
```

### 2. Mixed Data Types
The flags module supports different data types:
```ini
[FLAGS]
models = ['xgb', 'mlp']
features = ['os', 'praat']
learning_rate = [0.01, 0.1, 0.5]
epochs = [10, 50, 100]
```

### 3. Handling 'none' Values
When you specify `'none'` for balancing or scaling, the parameter won't be set in the configuration, using the default behavior:
```ini
[FLAGS]
balancing = ['none', 'smote']  # 'none' means no balancing parameter set
scale = ['none', 'standard']   # 'none' means no scaling parameter set
```

## Understanding the Output

When you run a flags experiment, you'll see output like this:

```
Flag parameters found: {'models': ['xgb', 'svm'], 'features': ['praat', 'os'], 'balancing': ['none', 'ros', 'smote'], 'scale': ['none', 'standard', 'robust', 'minmax']}
Running 48 experiment combinations...
Setting up experiment and extracting features (once for all experiments)...
Features extracted once: (1200, 384)

=== Experiment 1/48 ===
Parameters: {'models': 'xgb', 'features': 'praat', 'balancing': 'none', 'scale': 'none'}
Result: 0.67

=== Experiment 2/48 ===
Parameters: {'models': 'xgb', 'features': 'praat', 'balancing': 'none', 'scale': 'standard'}
Result: 0.71

...

=== SUMMARY OF 48 EXPERIMENTS ===
Experiment 1: {'models': 'xgb', 'features': 'praat', 'balancing': 'none', 'scale': 'none'}
  Result: 0.67
Experiment 2: {'models': 'xgb', 'features': 'praat', 'balancing': 'none', 'scale': 'standard'}
  Result: 0.71
...

=== BEST CONFIGURATION ===
Best Result: 0.84
Best Parameters:
  models: svm
  features: os
  balancing: smote
  scale: robust

To use these parameters, set in your config file:
[MODEL]
type = svm
[FEATS]
type = ['os']
balancing = smote
scale = robust

Flags experiments time: 245.67 seconds (4.09 minutes)
DONE
```

### Key Output Elements

1. **Parameter Discovery**: Shows all flag parameters found and their values
2. **Feature Extraction Info**: Displays the shape of extracted features (once for all experiments)
3. **Individual Results**: Each experiment shows its parameters and result score
4. **Comprehensive Summary**: Lists all experiments with their outcomes
5. **Best Configuration**: Identifies the highest-scoring parameter combination
6. **Usage Instructions**: Provides exact configuration syntax for the best parameters
7. **Timing Information**: Shows total execution time

### Result Interpretation

- **Result scores** represent the test performance metric (typically accuracy for classification)
- **Higher scores** indicate better performance
- **Feature extraction timing** is shown separately since it's done only once
- **Failed experiments** are marked with ERROR and don't affect other experiments

## Performance Optimization

The flags module includes several optimizations:

1. **Single Feature Extraction**: Features are extracted only once at the beginning and reused across all experiments
2. **Efficient Experiment Creation**: Each experiment reuses the base data and features rather than reloading
3. **Memory Optimization**: Configurations are generated on-the-fly rather than stored in memory

## Best Practices

### 1. Start Small
Begin with a smaller set of parameters to test the setup:
```ini
[FLAGS]
models = ['xgb']
features = ['os', 'praat']
balancing = ['none', 'smote']
```

### 2. Consider Computational Cost
Be mindful of the total number of combinations. With many parameters, the number of experiments can grow exponentially:
- 3 models × 4 features × 3 balancing × 4 scaling = 144 experiments

### 3. Use Meaningful Parameter Combinations
Not all parameter combinations make sense. For example, some scaling methods might not be beneficial for certain feature types.

### 4. Monitor Resources
Large flag experiments can be resource-intensive. Monitor CPU, memory, and disk usage, especially when working with large datasets.

## Troubleshooting

### Common Issues

1. **No FLAGS section error**: Ensure your configuration file has a `[FLAGS]` section
2. **Invalid parameter format**: Use proper Python list syntax with quotes: `['item1', 'item2']`
3. **Missing required sections**: Ensure your config file has all required sections (EXP, DATA, etc.)

### Error Handling
The flags module includes error handling for individual experiments. If one experiment fails, others will continue, and you'll see error information in the summary.

## Integration with Other Modules

The flags module can be combined with other nkululeko features:

### With Testing Module
```bash
python -m nkululeko.flags --config my_config.ini --mod test --flags
```

### With Custom Modules
The flags module supports different nkululeko modules through the `--mod` parameter:
- `nkulu`: Standard training (default)
- `test`: Testing mode

## Example Workflows

### 1. Feature Type Comparison
```ini
[FLAGS]
models = ['xgb']
features = ['os', 'praat', 'mfcc', 'hubert']
balancing = ['none']
scale = ['standard']
```

### 2. Model Selection
```ini
[FLAGS]
models = ['xgb', 'svm', 'mlp', 'tree']
features = ['os']
balancing = ['smote']
scale = ['standard']
```

### 3. Preprocessing Optimization
```ini
[FLAGS]
models = ['xgb']
features = ['os']
balancing = ['none', 'ros', 'smote', 'adasyn']
scale = ['none', 'standard', 'robust', 'minmax']
```

The flags module is a powerful tool for systematic experimentation in nkululeko, helping you find optimal configurations efficiently while maintaining reproducibility and comprehensive result tracking.

## Migration from nkuluflag

If you've used the legacy `nkuluflag` module, here are the key differences and migration steps:

### Key Improvements in flags module:

1. **Configuration-based**: Define parameters in INI files instead of command line only
2. **Optimized performance**: Features extracted once and reused across experiments  
3. **Better error handling**: Failed experiments don't stop the entire process
4. **Comprehensive output**: Clear summary with best configuration identification
5. **Flexible parameters**: Support for custom parameter types and prefixes

### Migration Steps:

**Old nkuluflag approach:**
```bash
python -m nkululeko.nkuluflag --config base.ini --model xgb svm --feat os praat --balancing none smote
```

**New flags approach:**
1. Add a `[FLAGS]` section to your INI file:
```ini
[FLAGS]
models = ['xgb', 'svm']
features = ['os', 'praat']
balancing = ['none', 'smote']
```

2. Run with the flags module:
```bash
python -m nkululeko.flags --config base.ini
```

The new approach is more maintainable, reproducible, and provides better performance for large parameter spaces.
