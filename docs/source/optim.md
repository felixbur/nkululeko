# Hyperparameter Optimization Module

The Nkululeko optimization module provides automated hyperparameter tuning for machine learning models. It helps you find the best parameter combinations to improve your model's performance without manual trial and error.

## Quick Start

### Basic Usage

```bash
# Set PYTHONPATH if running from source
export PYTHONPATH=/path/to/nkululeko

# Run optimization
python3 -m nkululeko.optim --config your_optimization_config.ini
```

### Minimal Example

Create a configuration file `simple_optim.ini`:

```ini
[EXP]
root = ./results/
name = simple_optimization
runs = 1
epochs = 1

[DATA]
databases = ['train', 'test']
train = ./data/your_data_train.csv
train.type = csv
train.split_strategy = train
test = ./data/your_data_test.csv
test.type = csv
test.split_strategy = test
target = emotion
labels = ['happy', 'sad', 'neutral']

[FEATS]
type = ['os']

[MODEL]
type = xgb

[OPTIM]
model = xgb
metric = uar
n_estimators = [50, 100]
max_depth = [3, 6]
learning_rate = [0.1, 0.2]
```

Run with:
```bash
python3 -m nkululeko.optim --config simple_optim.ini
```

## Optimization Approaches

The Nkululeko optimization module supports two main approaches to hyperparameter optimization:

### 1. Conventional Optimization
**Grid Search Exhaustive Exploration**
- Uses `search_strategy = grid` (default) or omits the strategy parameter
- Tests **all possible combinations** of parameters
- **Best for**: Small parameter spaces (< 100 combinations)
- **Guarantees**: Finding the optimal combination within the defined space
- **Trade-off**: Can be computationally expensive for large parameter spaces

**When to Use Conventional:**
- You have specific parameter values you want to test
- Parameter space is manageable (≤ 50-100 combinations)
- You want guaranteed comprehensive coverage
- Computational resources are not a limiting factor

### 2. Intelligent Optimization
**Smart Search Strategies**
- Uses advanced algorithms: `random`, `halving_grid`, `halving_random`
- **More efficient** for large parameter spaces
- Uses **statistical methods** to find good parameters faster
- **Best for**: Large parameter spaces (> 100 combinations)
- **Trade-off**: May not test all combinations but finds good solutions quickly

**Available Intelligent Strategies:**
- `random`: Random sampling from parameter distributions
- `halving_grid`: Successive halving with grid search (recommended)
- `halving_random`: Successive halving with random search

**When to Use Intelligent:**
- Large parameter spaces (> 100 combinations)
- Limited computational time or resources
- High-dimensional optimization problems
- You want to balance efficiency with effectiveness

## Real-World Examples with Polish Emotional Speech Dataset

The following examples demonstrate both conventional and intelligent optimization approaches using a real Polish emotional speech recognition dataset:

### Example 1: Conventional XGBoost Optimization

**Grid Search with Intelligent Enhancement**

This example demonstrates conventional optimization enhanced with `halving_grid` strategy for better efficiency:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_optim_xgb
runs = 1
epochs = 1
random_seed = 42

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']
scale = standard
balancing = smoteenn

[MODEL]
type = xgb
n_estimators = 50
max_depth = 9
learning_rate = 0.1
subsample = 0.9

[OPTIM]
model = xgb
search_strategy = halving_grid
n_iter = 15
cv_folds = 3
random_state = 42
n_estimators = [50, 100, 200]
max_depth = [3, 6, 9]
learning_rate = [0.01, 0.1, 0.2]
subsample = [0.8, 0.9, 1.0]
metric = uar
```

**Key features:**
- **Strategy**: `halving_grid` for intelligent search with grid coverage
- **Parameter space**: 3×3×3×3 = 81 combinations efficiently tested
- **Features**: Standard scaling + SMOTEENN balancing for class imbalance
- **Metric**: UAR (Unweighted Average Recall) for imbalanced emotion data
- **Reproducibility**: Fixed random seed for consistent results

### Example 2: Conventional SVM Grid Search

**Pure Grid Search Approach**

This example shows pure conventional grid search without intelligent enhancements:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_optim_svm
runs = 1
epochs = 1

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']
scale = robust
balancing = smoteenn

[MODEL]
type = svm
C_val = 0.1
kernel = linear

[OPTIM]
model = svm
C_val = [0.1, 1.0, 10.0, 100.0] 
kernel = ["linear", "rbf", "poly"]
gamma = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
metric = uar
```

**Key features:**
- **Strategy**: Default grid search (no `search_strategy` specified)
- **Parameter space**: 4×3×6 = 72 combinations tested exhaustively
- **Features**: Robust scaling + SMOTEENN balancing
- **Coverage**: Tests all parameter combinations systematically
- **Best for**: Medium-sized parameter spaces with guaranteed coverage

### Example 3: Intelligent SVM Optimization

**Advanced Halving Random Search**

This example demonstrates intelligent optimization for larger parameter spaces:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_optim_svm_intelligent
runs = 1
epochs = 1

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']

[MODEL]
type = svm

[OPTIM]
model = svm
search_strategy = halving_random
n_iter = 15
cv_folds = 3
C_val = [0.1, 1.0, 10.0, 100.0, 1000.0]
kernel = ["linear", "rbf", "poly"]
gamma = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
metric = uar
```

**Key features:**
- **Strategy**: `halving_random` for maximum efficiency
- **Parameter space**: 5×3×6 = 90 combinations, intelligently sampled
- **Efficiency**: Only tests most promising combinations using successive halving
- **Iterations**: Limited to 15 iterations for time efficiency
- **Best for**: Large parameter spaces with time constraints

### Example 4: Neural Network (MLP) Conventional Optimization

**Grid Search for Deep Learning**

This example shows conventional grid search for neural network hyperparameters:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_optim_mlp
runs = 1
epochs = 5

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']

[OPTIM]
model = mlp
nlayers = [1, 2]
nnodes = [16, 32]
lr = [0.0001, 0.001]
bs = [4, 8, 16, 32]
loss = ["cross", "f1"]
do = [0.1, 0.3, 0.5]
metric = uar
```

**Key features:**
- **Strategy**: Default grid search for systematic exploration
- **Parameter space**: 2×2×2×4×2×3 = 192 combinations
- **Architecture**: Tests both network structure (layers, nodes) and training parameters
- **Training**: Includes learning rate, batch size, dropout, and loss function
- **Comprehensive**: Covers all major neural network hyperparameters

## Choosing the Right Approach

### When to Use Conventional Optimization

**Ideal scenarios:**
- Parameter space ≤ 100 combinations
- You have specific parameter values to test
- Computational resources are abundant
- You need guaranteed coverage of all combinations
- You're fine-tuning around known good parameters

**Example parameter spaces for conventional:**
```ini
# Small XGBoost optimization (3×3×2 = 18 combinations)
n_estimators = [50, 100, 200]
max_depth = [3, 6, 9]
learning_rate = [0.1, 0.2]

# Small SVM optimization (3×2×3 = 18 combinations)
C_val = [1.0, 10.0, 100.0]
kernel = ["linear", "rbf"]
gamma = [0.001, 0.01, 0.1]
```

### When to Use Intelligent Optimization

**Ideal scenarios:**
- Parameter space > 100 combinations
- Limited computational time or resources
- High-dimensional parameter spaces
- Exploring wide parameter ranges
- Initial parameter exploration

**Example parameter spaces for intelligent:**
```ini
# Large XGBoost optimization (4×4×4×3 = 192 combinations)
n_estimators = [50, 100, 200, 500]
max_depth = [3, 6, 9, 12]
learning_rate = [0.01, 0.05, 0.1, 0.2]
subsample = [0.7, 0.8, 0.9]

# With halving_grid, only tests ~50-60 combinations intelligently
```

### Strategy Comparison

| Strategy | Parameter Space Size | Time Efficiency | Coverage Guarantee | Best Use Case |
|----------|---------------------|-----------------|-------------------|---------------|
| `grid` | < 100 combinations | Low | 100% | Small, specific searches |
| `random` | Any size | High | Statistical | Large spaces, quick exploration |
| `halving_grid` | > 50 combinations | Medium-High | High | Balanced efficiency and coverage |
| `halving_random` | > 100 combinations | Highest | Medium | Very large spaces, time-limited |

This example demonstrates conventional grid search for SVM hyperparameters:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_optim_svm
runs = 1
epochs = 1

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']
scale = robust
balancing = smoteenn

[MODEL]
type = svm
C_val = 0.1
kernel = linear

[OPTIM]
model = svm
C_val = [0.1, 1.0, 10.0, 100.0] 
kernel = ["linear", "rbf", "poly"]
gamma = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
metric = uar
```

**Key features of this example:**
- Uses default `grid` search strategy (conventional)
- Tests 4×3×6 = 72 parameter combinations
- Combines different parameter types (numerical and categorical)
- Uses robust scaling for features
- No explicit search strategy means conventional grid search

### Example 3: Neural Network (MLP) Optimization

This example shows optimization for Multi-Layer Perceptron networks:

```ini
[EXP]
root = ./examples/results/
name = exp_polish_optim_mlp
runs = 1
epochs = 5

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']

[OPTIM]
model = mlp
nlayers = [1, 2]
nnodes = [16, 32]
lr = [0.0001, 0.001]
bs = [4, 8, 16, 32]
loss = ["cross", "f1"]
do = [0.1, 0.3, 0.5]
metric = uar
```

**Key features of this example:**
- Optimizes neural network architecture and training parameters
- Tests 2×2×2×4×2×3 = 192 parameter combinations
- Includes architectural parameters (layers, nodes)
- Includes training parameters (learning rate, batch size, dropout)
- Includes loss function selection
- Uses conventional grid search for comprehensive evaluation

## Configuration Parameters

### Core Settings

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model` | Model type to optimize | `mlp` | `mlp`, `xgb`, `svm`, `knn`, `tree`, etc. |
| `search_strategy` | Search method | `grid` | `grid`, `random`, `halving_grid`, `halving_random` |
| `metric` | Optimization metric | `accuracy` | `uar`, `accuracy`, `f1`, `precision`, `recall` |
| `n_iter` | Number of iterations (for random search) | `50` | Any integer |
| `cv_folds` | Cross-validation folds | `3` | Any integer ≥ 2 |
| `random_state` | Random seed for reproducibility | `42` | Any integer |

### Search Strategies

#### Grid Search (`grid`)
- **Best for**: Small parameter spaces (< 100 combinations)
- **Pros**: Exhaustive search, guaranteed to find the best combination
- **Cons**: Computationally expensive for large parameter spaces

#### Random Search (`random`)
- **Best for**: Large parameter spaces, limited time budget
- **Pros**: More efficient than grid search for high-dimensional spaces
- **Cons**: May miss optimal combinations

#### Halving Grid Search (`halving_grid`)
- **Best for**: Large parameter spaces with successive halving
- **Pros**: More efficient than regular grid search
- **Cons**: Requires scikit-learn >= 0.24

#### Halving Random Search (`halving_random`)
- **Best for**: Very large parameter spaces
- **Pros**: Combines benefits of random search with successive halving
- **Cons**: Most complex, may need tuning

## Parameter Specification

### List Format
Specify discrete values to test:
```ini
C_val = [0.1, 1.0, 10.0, 100.0]
kernel = ["linear", "rbf", "poly"]
```

### Range Format
For continuous parameters (automatically generates reasonable steps):
```ini
# Creates range with smart step selection
learning_rate = (0.001, 0.1)  # Will generate log-spaced values
max_depth = (3, 10)           # Will generate integer range
```

### Range with Step
Specify exact step size:
```ini
# (min, max, step)
dropout = (0.1, 0.5, 0.1)     # [0.1, 0.2, 0.3, 0.4, 0.5]
```

## Model-Specific Parameters

### XGBoost (`xgb`, `xgr`)
```ini
[OPTIM]
model = xgb
n_estimators = [50, 100, 200]
max_depth = [3, 6, 9, 12]
learning_rate = [0.01, 0.1, 0.3]
subsample = [0.6, 0.8, 1.0]
colsample_bytree = [0.6, 0.8, 1.0]
```

### Support Vector Machine (`svm`, `svr`)
```ini
[OPTIM]
model = svm
C_val = [0.1, 1.0, 10.0, 100.0]
kernel = ["linear", "rbf", "poly"]
gamma = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
```

### K-Nearest Neighbors (`knn`, `knn_reg`)
```ini
[OPTIM]
model = knn
K_val = [3, 5, 7, 9, 11]
weights = ["uniform", "distance"]
algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
```

### Multi-Layer Perceptron (`mlp`)
```ini
[OPTIM]
model = mlp
nlayers = [1, 2, 3]
nnodes = [16, 32, 64, 128]
lr = [0.0001, 0.001, 0.01]
bs = [8, 16, 32, 64]
do = [0.1, 0.3, 0.5]
loss = ["cross", "f1", "mse"]
```

## Complete Examples

### Example 1: XGBoost Optimization
```ini
[EXP]
root = ./results/
name = exp_polish_optim_xgb
runs = 1
epochs = 1

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']
scale = standard
balancing = smoteenn

[MODEL]
type = xgb
n_estimators = 50
max_depth = 6
learning_rate = 0.1

[OPTIM]
model = xgb
search_strategy = halving_grid
n_iter = 15
cv_folds = 3
random_state = 42
n_estimators = [50, 100, 200]
max_depth = [3, 6, 9]
learning_rate = [0.01, 0.1, 0.2]
subsample = [0.8, 0.9, 1.0]
metric = uar
```

### Example 2: SVM Optimization
```ini
[EXP]
root = ./results/
name = exp_polish_optim_svm
runs = 1
epochs = 1

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']
scale = robust
balancing = smoteenn

[MODEL]
type = svm
C_val = 1.0
kernel = rbf

[OPTIM]
model = svm
search_strategy = grid
C_val = [0.1, 1.0, 10.0, 100.0]
kernel = ["linear", "rbf", "poly"]
gamma = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
metric = uar
```

### Example 3: Neural Network (MLP) Optimization
```ini
[EXP]
root = ./results/
name = exp_polish_optim_mlp
runs = 1
epochs = 5

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
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']

[OPTIM]
model = mlp
search_strategy = random
n_iter = 20
cv_folds = 3
metric = uar
nlayers = [1, 2]
nnodes = [16, 32, 64]
lr = [0.0001, 0.001, 0.01]
bs = [8, 16, 32]
loss = ["cross", "f1"]
do = [0.1, 0.3, 0.5]
```

## Understanding Results

### Output Files
The optimization process creates several output files in your experiment directory:

- `optimization_results_{model}.csv`: Detailed results for all parameter combinations
- `images/`: Visualization plots (if enabled)
- `results/`: Text-based result summaries

### Result Interpretation
The optimization will output:
- **Best parameters**: The parameter combination that achieved the highest score
- **Best score**: The performance metric value for the best parameters
- **All results**: Complete results table with all tested combinations

### Cross-Validation vs. Final Evaluation
The module provides warnings when there are large discrepancies between cross-validation scores and final test evaluation. This helps identify potential overfitting issues.

## Advanced Features

### Consistency Improvements
The optimization module includes several features to ensure consistent results:

- **Stratified Cross-Validation**: Maintains class distribution across CV folds
- **Consistent Data Balancing**: Applies the same balancing strategy used in final evaluation
- **Reproducible Results**: Fixed random seeds for consistent results
- **Validation Checks**: Compares CV results with standard evaluation pipeline

### Performance Optimization
- **Parallel Processing**: Uses multiple CPU cores when available
- **Early Stopping**: Halving strategies reduce computation time
- **Smart Parameter Ranges**: Automatic generation of reasonable parameter ranges

## Best Practices

### 1. Start Small
Begin with a small parameter space to understand your model's behavior:
```ini
# Start with 2-3 values per parameter
C_val = [1.0, 10.0]
kernel = ["linear", "rbf"]
```

### 2. Use Appropriate Search Strategies
- **Grid search**: For ≤ 50 parameter combinations
- **Random search**: For > 50 combinations
- **Halving methods**: For > 200 combinations

### 3. Choose the Right Metric
- **Classification**: `uar` (Unweighted Average Recall) for imbalanced datasets
- **Balanced datasets**: `accuracy` or `f1`
- **Regression**: `mse`, `mae`, or `r2`

### 4. Set Reasonable CV Folds
- **Small datasets (< 1000 samples)**: 3-5 folds
- **Large datasets**: 5-10 folds
- **Very small datasets**: Consider leave-one-out CV

### 5. Monitor for Overfitting
Watch for large discrepancies between CV and test scores:
```
Cross-validation score: 0.8500
Standard evaluation score: 0.7200
Score difference: 0.1300
WARNING: Large discrepancy detected!
```

## Troubleshooting

### Common Issues

#### 1. No Parameter Combinations Generated
**Problem**: Empty parameter space
**Solution**: Check parameter syntax in the `[OPTIM]` section

#### 2. Memory Issues
**Problem**: Too many parameter combinations
**Solution**: Reduce parameter space or use random/halving search

#### 3. Slow Optimization
**Problem**: Long execution time
**Solution**: 
- Use fewer CV folds
- Reduce parameter space
- Use halving strategies

#### 4. Poor Results
**Problem**: Optimized parameters perform worse than defaults
**Solutions**:
- Check for data leakage
- Ensure consistent preprocessing
- Verify parameter ranges are reasonable

### Error Messages

#### "No [OPTIM] section found"
Add an `[OPTIM]` section to your configuration file.

#### "Large discrepancy between CV and standard evaluation"
This indicates potential overfitting or data inconsistency. Consider:
- Reducing model complexity
- Checking data preprocessing steps
- Using simpler search strategies

## Performance Tips

### 1. Efficient Parameter Ranges
Use logarithmic ranges for parameters that span multiple orders of magnitude:
```ini
# Good for learning rates
lr = [0.0001, 0.001, 0.01, 0.1]

# Instead of linear spacing
# lr = [0.0001, 0.0002, 0.0003, ..., 0.1]  # Too many values
```

### 2. Use Intelligent Defaults
Start with recommended parameter ranges:
```ini
# XGBoost recommended ranges
n_estimators = [50, 100, 200]        # Not [10, 20, 30, ..., 1000]
max_depth = [3, 6, 9]                # Not [1, 2, 3, ..., 20]
learning_rate = [0.01, 0.1, 0.3]     # Not [0.001, 0.002, ..., 1.0]
```

### 3. Parallel Processing
The module automatically uses multiple cores. For better performance:
- Use machines with more CPU cores
- Ensure sufficient RAM for parallel processes
- Consider using smaller batch sizes for neural networks

## Integration with Nkululeko Workflow

The optimization module integrates seamlessly with the standard Nkululeko workflow:

1. **Setup**: Same data and feature configuration as regular experiments
2. **Optimization**: Add `[OPTIM]` section and run optimization
3. **Final Training**: Use best parameters in a standard experiment
4. **Evaluation**: Compare optimized vs. default performance

### After Optimization
Once you find the best parameters, update your model configuration:
```ini
[MODEL]
type = xgb
n_estimators = 200    # From optimization results
max_depth = 6         # From optimization results
learning_rate = 0.1   # From optimization results
subsample = 0.9       # From optimization results
```

Then run a standard experiment to get final results:
```bash
python3 -m nkululeko.nkululeko --config your_final_config.ini
```

## Summary

The Nkululeko optimization module provides a powerful and flexible framework for hyperparameter tuning. Key benefits include:

- **Automated parameter search** with multiple strategies
- **Consistent evaluation** pipeline to avoid overfitting
- **Support for all model types** in Nkululeko
- **Detailed result analysis** and performance monitoring
- **Best practice recommendations** built-in

Start with simple grid searches and gradually move to more sophisticated methods as your parameter spaces grow. Always validate your results with independent test sets and watch for overfitting indicators.
