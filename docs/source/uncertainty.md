# Using Uncertainty in Predictions

This tutorial explains how to use uncertainty (model confidence) visualization and thresholding in Nkululeko. Uncertainty helps you understand when your model is confident about its predictions and when it's unsure.

## Overview

Since Nkululeko version 0.94, aleatoric uncertainty (model confidence) is explicitly visualized. After running an experiment, you'll find an uncertainty distribution plot in the `images` folder showing how uncertainty correlates with prediction accuracy.

## What is Uncertainty?

In classification, **uncertainty** measures how confident the model is about its prediction. It's typically computed using **entropy** of the predicted probability distribution:

- **Low uncertainty**: Model is confident (probabilities concentrated on one class)
- **High uncertainty**: Model is unsure (probabilities spread across multiple classes)

## Automatic Uncertainty Visualization

When you run any classification experiment, Nkululeko automatically generates an uncertainty distribution plot:

```
results/<exp_name>/images/uncertainty_distribution.png
```

This plot shows:
- Distribution of uncertainty values for **correct predictions** (usually lower uncertainty)
- Distribution of uncertainty values for **incorrect predictions** (usually higher uncertainty)

A well-calibrated model shows clear separation between these distributions.

## Using Uncertainty Threshold

You can use uncertainty to **filter out low-confidence predictions**. This is useful when:
- It's better to give no prediction than a wrong one
- You're working with critical applications (medical, safety)
- You want to identify samples that need human review

### Configuration

Add the `uncertainty_threshold` option in the `[PLOT]` section:

```ini
[PLOT]
uncertainty_threshold = 0.4
```

This will:
1. Generate the standard confusion matrix
2. Generate an **additional confusion matrix** excluding samples above the threshold
3. Show how accuracy improves when uncertain samples are filtered

### Example Configuration: `exp_emodb_uncertainty.ini`

```ini
[EXP]
root = ./examples/results/
name = exp_emodb_uncertainty
runs = 1
epochs = 1
save = True

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
type = xgb

[PLOT]
# Uncertainty threshold: refuse to predict samples above this entropy value
# Lower values = stricter filtering (more samples rejected)
# Higher values = more permissive (fewer samples rejected)
uncertainty_threshold = 0.4
```

### Run the Experiment

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_uncertainty.ini
```

## Choosing the Right Threshold

| Threshold | Effect |
|-----------|--------|
| `0.2` | Very strict - only very confident predictions |
| `0.4` | Moderate - balanced filtering |
| `0.6` | Permissive - most predictions included |
| `0.8` | Very permissive - almost all predictions |

The optimal threshold depends on your use case:
- **High-stakes applications**: Use lower threshold (e.g., 0.2-0.3)
- **General analysis**: Use moderate threshold (e.g., 0.4-0.5)
- **Maximum coverage**: Use higher threshold or no filtering

## Output Files

After running with `uncertainty_threshold`, you'll find:

```
results/exp_emodb_uncertainty/
├── images/
│   ├── confusion_matrix.png          # Standard confusion matrix
│   ├── confusion_matrix_filtered.png # Filtered by uncertainty
│   └── uncertainty_distribution.png  # Uncertainty histogram
└── results/
    └── results.csv                   # Includes uncertainty metrics
```

## Interpreting Results

### Uncertainty Distribution Plot

A good model shows:
- **Correct predictions** clustered at **low uncertainty** (left side)
- **Incorrect predictions** spread toward **high uncertainty** (right side)
- Clear separation between the two distributions

### Filtered Confusion Matrix

Compare the filtered and unfiltered confusion matrices:
- **Improved accuracy**: Filtering removes uncertain (often wrong) predictions
- **Reduced samples**: Fewer samples in the filtered matrix
- **Trade-off**: Better accuracy vs. fewer predictions

## Use Cases

1. **Quality control**: Identify samples the model struggles with
2. **Active learning**: Select uncertain samples for human labeling
3. **Cascaded systems**: Route uncertain samples to more powerful models
4. **Safety-critical applications**: Refuse to predict when unsure
5. **Model debugging**: Understand where the model lacks confidence

## Practical Example

Consider a medical diagnosis system:
- Without threshold: 85% accuracy on all samples
- With `uncertainty_threshold = 0.3`: 95% accuracy on 70% of samples

In this case, 30% of samples are flagged for human review, but the automatic predictions are much more reliable.

## Combining with Other Features

Uncertainty works well with:
- **Multiple runs** (`runs = 5`): Get uncertainty estimates across runs
- **Ensemble models**: Combine predictions from multiple models
- **Feature importance**: Identify which features cause uncertainty

## Tips

1. **Start with moderate threshold** (0.4) and adjust based on results
2. **Monitor coverage**: Check what percentage of samples pass the threshold
3. **Analyze uncertain samples**: They often reveal data quality issues
4. **Use with test data**: Evaluate on held-out data for realistic estimates

## Related Resources

- Paper: [Uncertainty-Based Ensemble Learning For Speech Classification](https://arxiv.org/abs/2407.17009)
- [Visualization Guide](visualization.md)
- [Plots Reference](plots.md)

## Reference

- [Blog: Nkululeko - Using Uncertainty](http://blog.syntheticspeech.de/2025/08/04/nkululeko-using-uncertainty/)
