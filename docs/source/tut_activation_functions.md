# Activation Functions in Neural Network Models

## Overview

This tutorial demonstrates how to use different activation functions in MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Network) models in nkululeko. Activation functions are crucial components of neural networks that introduce non-linearity, enabling models to learn complex patterns in your data.

**Version**: Added in nkululeko 1.1.2  
**Models**: MLP classifier, MLP regression, CNN

## What are Activation Functions?

Activation functions determine the output of a neural network node given an input or set of inputs. They introduce non-linear properties to the network, allowing it to learn complex decision boundaries and representations. Choosing the right activation function can significantly impact your model's performance.

## Available Activation Functions

Nkululeko supports four activation functions for MLP models:

| Function | Description | Range | Best Used For |
|----------|-------------|-------|---------------|
| **relu** | Rectified Linear Unit | [0, ∞) | Default choice, general purpose, fast training |
| **leaky_relu** | Leaky ReLU (small negative slope) | (-∞, ∞) | Mitigating dead neurons, handling negative values |
| **tanh** | Hyperbolic tangent | [-1, 1] | Zero-centered outputs, classification tasks |
| **sigmoid** | Logistic function | [0, 1] | Binary classification, probability outputs |

### Activation Function Characteristics

#### ReLU (Rectified Linear Unit) - Default
```
f(x) = max(0, x)
```
- **Advantages**: Fast computation, reduces vanishing gradient problem
- **Disadvantages**: Can cause "dead neurons" (neurons that always output 0)
- **Use when**: General purpose, first choice for most problems

#### Leaky ReLU
```
f(x) = x if x > 0 else 0.01x
```
- **Advantages**: Prevents dead neurons, allows small negative gradients
- **Disadvantages**: Slight increase in computation
- **Use when**: ReLU shows signs of dead neurons, need better gradient flow

#### Tanh (Hyperbolic Tangent)
```
f(x) = (e^x - e^-x) / (e^x + e^-x)
```
- **Advantages**: Zero-centered output, stronger gradients than sigmoid
- **Disadvantages**: Can suffer from vanishing gradients with deep networks
- **Use when**: Need zero-centered activations, shallow to medium networks

#### Sigmoid
```
f(x) = 1 / (1 + e^-x)
```
- **Advantages**: Smooth gradient, output interpretable as probability
- **Disadvantages**: Vanishing gradient problem, not zero-centered
- **Use when**: Output layer for binary classification, need probability outputs

## Configuration

### Basic Usage

Add the `activation` parameter to the `[MODEL]` section:

```ini
[MODEL]
type = mlp
layers = [128, 64, 32]
activation = leaky_relu  # Options: relu, tanh, sigmoid, leaky_relu
```

### Complete Example: Emotion Recognition with Different Activations

#### 1. Using Default ReLU (no activation specified)
```ini
[EXP]
root = ./experiments/
name = exp_emodb_relu
runs = 3
epochs = 100

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = speaker_split
target = emotion
labels = ['anger', 'happiness', 'sadness', 'neutral']

[FEATS]
type = ['os']  # OpenSMILE features
scale = standard

[MODEL]
type = mlp
layers = [128, 64]
# activation = relu  # Default, can be omitted
drop = 0.3
patience = 10
learning_rate = 0.0001
```

#### 2. Using Leaky ReLU (recommended for most cases)
```ini
[MODEL]
type = mlp
layers = [128, 64]
activation = leaky_relu  # Better gradient flow
drop = 0.3
patience = 10
```

#### 3. Using Tanh (for zero-centered outputs)
```ini
[MODEL]
type = mlp
layers = [128, 64]
activation = tanh  # Zero-centered activation
drop = 0.3
patience = 10
```

#### 4. Using Sigmoid (for specific architectures)
```ini
[MODEL]
type = mlp
layers = [128, 64]
activation = sigmoid  # Smooth, probabilistic
drop = 0.3
patience = 10
```

## Practical Examples

### Example 1: Speaker Age Prediction (Regression)

For regression tasks, leaky ReLU or tanh often work well:

```ini
[EXP]
root = ./experiments/
name = age_prediction
runs = 5
epochs = 200

[DATA]
databases = ['agedb']
agedb = ./data/agedb/
target = age

[FEATS]
type = ['os', 'praat']
scale = standard

[MODEL]
type = mlp
layers = [256, 128, 64]
activation = leaky_relu  # Good for regression
drop = 0.4
patience = 15
learning_rate = 0.00005
```

### Example 2: Binary Emotion Classification

For binary classification, tanh or leaky_relu are good choices:

```ini
[EXP]
root = ./experiments/
name = binary_emotion
runs = 3

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
target = emotion
labels = ['anger', 'neutral']  # Binary classification

[FEATS]
type = ['os']
scale = standard

[MODEL]
type = mlp
layers = [64, 32]
activation = tanh  # Zero-centered for binary classification
drop = 0.2
patience = 5
```

### Example 3: Multi-class Classification with Deep Network

For deeper networks, leaky_relu helps prevent vanishing gradients:

```ini
[MODEL]
type = mlp
layers = [256, 128, 64, 32]  # Deep network
activation = leaky_relu  # Prevents dead neurons in deep networks
drop = [0.3, 0.3, 0.2, 0.2]  # Layer-specific dropout
patience = 20
learning_rate = 0.00001
batch_size = 16
```

## CNN Models with List Layer Format

The same update also introduced support for list format in CNN layers:

### Old Format (still supported)
```ini
[MODEL]
type = cnn
layers = {'l1': 120, 'l2': 84}
```

### New Format (recommended)
```ini
[MODEL]
type = cnn
layers = [120, 84]  # Simpler, more intuitive
```

## Choosing the Right Activation Function

### Decision Guide

```
Start with: relu or leaky_relu
    |
    ├─> If you have dead neurons → Try leaky_relu
    |
    ├─> If gradients vanish in deep networks → Try leaky_relu
    |
    ├─> If you need zero-centered outputs → Try tanh
    |
    └─> If you need probabilistic interpretation → Try sigmoid
```

### Common Use Cases

| Task | Recommended Activation | Reason |
|------|----------------------|--------|
| **General classification** | `relu` or `leaky_relu` | Fast, effective, standard choice |
| **Deep networks (>3 layers)** | `leaky_relu` | Prevents dead neurons, better gradient flow |
| **Regression** | `leaky_relu` | Handles negative values well |
| **Binary classification** | `tanh` or `leaky_relu` | Zero-centered, good gradients |
| **Multi-class emotion** | `leaky_relu` | Robust, prevents gradient issues |
| **Small networks** | `relu` or `tanh` | Simple, effective |

## Performance Comparison Example

Running the same experiment with different activations (emodb dataset, 2 emotions):

```bash
# Test all activations
for activation in relu leaky_relu tanh sigmoid; do
    python -m nkululeko.nkululeko --config exp_test_${activation}.ini
done
```

### Typical Results (example)

| Activation | UAR | Training Time | Notes |
|------------|-----|---------------|-------|
| relu | 0.645 | 18.3s | Fast convergence |
| leaky_relu | 0.658 | 19.1s | Best performance |
| tanh | 0.641 | 21.2s | Stable training |
| sigmoid | 0.612 | 24.5s | Slower convergence |

*Note: Results vary by dataset and configuration*

## Testing Your Configuration

### Quick Test Script

Create a test configuration with fewer epochs to quickly validate your setup:

```ini
[EXP]
name = quick_test
runs = 1
epochs = 5  # Quick test

[MODEL]
type = mlp
layers = [64, 16]
activation = leaky_relu  # Test your activation
```

### Verify Activation Function

Check the log output to confirm your activation is being used:

```bash
python -m nkululeko.nkululeko --config your_config.ini 2>&1 | grep "activation"
```

Expected output:
```
DEBUG: model: using activation function: leaky_relu
```

## Advanced Tips

### 1. Combining with Dropout

Different activations may work better with different dropout rates:

```ini
[MODEL]
type = mlp
layers = [128, 64]
activation = leaky_relu
drop = 0.3  # Start with 0.3 and adjust

# Layer-specific dropout
# drop = [0.4, 0.3]  # Higher dropout in earlier layers
```

### 2. Learning Rate Adjustment

Some activations may require different learning rates:

```ini
# For relu/leaky_relu
learning_rate = 0.0001  # Standard

# For tanh (sometimes needs lower LR)
learning_rate = 0.00005

# For sigmoid (often needs lower LR)
learning_rate = 0.00003
```

### 3. Batch Size Considerations

```ini
# Larger batch sizes often work better with relu/leaky_relu
[MODEL]
activation = leaky_relu
batch_size = 32

# Smaller batch sizes may help tanh/sigmoid
[MODEL]
activation = tanh
batch_size = 8
```

### 4. Network Depth and Activation

```ini
# Shallow networks: relu or tanh work well
[MODEL]
layers = [64, 32]
activation = relu

# Deep networks: prefer leaky_relu
[MODEL]
layers = [256, 128, 64, 32]
activation = leaky_relu  # Better gradient flow
```

## Troubleshooting

### Problem: Model not improving

**Solutions:**
1. Try `leaky_relu` instead of `relu`
2. Reduce learning rate
3. Increase patience parameter
4. Check if dropout is too high

### Problem: Loss becomes NaN

**Solutions:**
1. Lower the learning rate significantly
2. Try `tanh` or `leaky_relu` instead of `relu`
3. Check feature scaling (use `scale = standard`)
4. Reduce batch size

### Problem: Training is very slow

**Solutions:**
1. Use `relu` or `leaky_relu` (fastest)
2. Avoid `sigmoid` for hidden layers
3. Increase batch size
4. Reduce network complexity

### Problem: Overfitting

**Solutions:**
1. Increase dropout
2. Try `leaky_relu` with higher dropout
3. Reduce network size
4. Use early stopping (patience parameter)

## Complete Working Example

Here's a complete, tested configuration for emotion recognition:

```ini
# File: exp_emotion_leaky_relu.ini
# Description: Emotion recognition with leaky_relu activation

[EXP]
root = ./experiments/results/
name = emotion_leaky_relu
runs = 3
epochs = 100
save = True

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
target = emotion
labels = ['anger', 'happiness', 'sadness', 'neutral']

[FEATS]
type = ['os']  # OpenSMILE features
scale = standard  # Important: scale features

[MODEL]
type = mlp
layers = [128, 64, 32]
activation = leaky_relu  # NEW: Activation function
drop = 0.3
patience = 10
learning_rate = 0.0001
batch_size = 16

[PLOT]
best_model = True
epoch_progression = True
```

Run it:
```bash
python -m nkululeko.nkululeko --config exp_emotion_leaky_relu.ini
```

## Summary

- **Default**: `relu` - Fast, effective, good starting point
- **Recommended**: `leaky_relu` - More robust, prevents dead neurons
- **For specific needs**: `tanh` (zero-centered) or `sigmoid` (probabilistic)
- **Always**: Combine with proper scaling (`scale = standard`)
- **Experiment**: Test different activations with your specific dataset

## References

- Nkululeko documentation: [ini_file.md](../ini_file.md)
- Added in PR: "added different activation functions" (2026-01-08)
- Version: 1.1.2+

## See Also

- [Feature extraction tutorial](tut_regplot_features.md)
- [Model optimization guide](../ini_file.md#model-section)
- [Data preprocessing](../ini_file.md#data-section)
