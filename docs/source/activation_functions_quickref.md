# Activation Functions Quick Reference

## Quick Configuration

```ini
[MODEL]
type = mlp
layers = [128, 64]
activation = leaky_relu  # Options: relu, tanh, sigmoid, leaky_relu
```

## Available Functions

| Function | Config Value | Use Case |
|----------|--------------|----------|
| ReLU | `relu` | Default, general purpose |
| Leaky ReLU | `leaky_relu` | **Recommended** - prevents dead neurons |
| Tanh | `tanh` | Zero-centered outputs |
| Sigmoid | `sigmoid` | Probabilistic outputs |

## Quick Decision Tree

```
Choose:
├─ General use? → leaky_relu
├─ Deep network? → leaky_relu
├─ Regression? → leaky_relu
├─ Need zero-centered? → tanh
└─ Binary probability? → sigmoid
```

## Example Configurations

### Classification
```ini
[MODEL]
type = mlp
layers = [128, 64, 32]
activation = leaky_relu
drop = 0.3
patience = 10
```

### Regression
```ini
[MODEL]
type = mlp
layers = [256, 128, 64]
activation = leaky_relu
drop = 0.4
patience = 15
```

### CNN with List Layers
```ini
[MODEL]
type = cnn
layers = [120, 84]  # New list format!
patience = 5
```

## Test Your Setup

```bash
# Quick test with 5 epochs
python -m nkululeko.nkululeko --config your_config.ini

# Check activation is loaded
python -m nkululeko.nkululeko --config your_config.ini 2>&1 | grep activation
# Should see: DEBUG: model: using activation function: leaky_relu
```

## Full Tutorial

See [tut_activation_functions.md](tut_activation_functions.md) for complete documentation.
