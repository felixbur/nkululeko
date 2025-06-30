# Quick Start Example for Nkululeko Optimization

This example demonstrates how to use the Nkululeko optimization module to find the best hyperparameters for your machine learning model.

## Example Configuration

Create a file called `my_optimization.ini`:

```ini
[EXP]
root = ./results/
name = my_optimization_experiment
runs = 1
epochs = 1

[DATA]
databases = ['train', 'test']
train = ./data/your_train_data.csv
train.type = csv
train.split_strategy = train
test = ./data/your_test_data.csv
test.type = csv  
test.split_strategy = test
target = emotion
labels = ['happy', 'sad', 'angry']

[FEATS]
type = ['os']
scale = standard

[MODEL]
type = xgb

[OPTIM]
model = xgb
search_strategy = grid
metric = uar
n_estimators = [50, 100]
max_depth = [3, 6]
learning_rate = [0.1, 0.2]
```

## Running the Optimization

```bash
# Run optimization with your configuration
PYTHONPATH=/path/to/nkululeko python3 -m nkululeko.optim --config my_optimization.ini

# Or if nkululeko is installed
python3 -m nkululeko.optim --config my_optimization.ini
```

## Understanding the Output

The optimization will test all combinations:
- 2 values for n_estimators × 2 values for max_depth × 2 values for learning_rate = 8 total combinations

Results will be saved in:
- `./results/my_optimization_experiment/optimization_results_xgb.csv`

## Best Practices

1. **Start small**: Begin with 2-3 values per parameter
2. **Use appropriate metrics**: 
   - `uar` for imbalanced datasets
   - `accuracy` for balanced datasets
3. **Choose the right search strategy**:
   - `grid` for small parameter spaces (< 50 combinations)
   - `random` for larger spaces
   - `halving_grid` or `halving_random` for very large spaces

## Next Steps

After optimization completes:
1. Note the best parameters from the output
2. Update your model configuration with these parameters
3. Run a final experiment to validate the results
