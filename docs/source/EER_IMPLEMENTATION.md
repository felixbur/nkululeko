# Equal Error Rate (EER) Implementation for Nkululeko

## Overview

This implementation adds Equal Error Rate (EER) as a metric for binary classification tasks in nkululeko, addressing issue in [https://github.com/bagustris/nkululeko/issues/15](https://github.com/bagustris/nkululeko/issues/15). EER is particularly useful for biometric systems and deepfake detection tasks like the Fake-or-Real (FoR) dataset.

## What is EER?

Equal Error Rate (EER) is the point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR) in a binary classification system. It's commonly used in:
- Biometric authentication systems
- Speaker verification
- Deepfake/synthetic speech detection
- Security systems

**Lower EER values indicate better performance** (range: 0-1, where 0 is perfect).

## Implementation Details

### Files Modified

1. **nkululeko/reporting/reporter.py**
   - Added `equal_error_rate()` function
   - Updated `_set_metric()` to support EER
   - Modified `_get_test_result()` to calculate EER
   - Updated `__init__()` to compute both EER and UAR when EER is selected
   - Modified plotting functions to display both EER and UAR

2. **ini_file.md**
   - Updated documentation for the `measure` parameter
   - Added EER as an option for classification tasks

### New Files Created

1. **tests/test_eer.py**
   - Unit tests for EER calculation
   - Validates EER with different classification scenarios

2. **data/for-2sec/exp_eer.ini**
   - Example configuration using EER metric
   - Demonstrates usage with FoR-2sec deepfake detection dataset

## Usage

### In Configuration Files

Add the following to your INI file's `[MODEL]` section:

```ini
[MODEL]
type = xgb  # or any classifier
measure = eer
```

### Example

See [data/for-2sec/exp_eer.ini](../data/for-2sec/exp_eer.ini) for a complete example:

```bash
python -m nkululeko.nkululeko --config data/for-2sec/exp_eer.ini
```

## Key Features

1. **Dual Reporting**: When EER is selected as the measure, both EER and UAR are reported
2. **Confidence Intervals**: EER is calculated with bootstrap confidence intervals (same as other metrics)
3. **Probability-Based**: EER uses class probabilities when available for accurate calculation
4. **Fallback Handling**: Gracefully handles cases where probabilities are not available

## Output Format

When using EER, the output will show:
```
Confusion matrix result for epoch: 0, EER: 0.123, (+-0.015/0.018), UAR: 0.876, ACC: 0.892
```

Confusion matrix plots will display both EER and UAR in the title.

## Testing

Run the unit tests:
```bash
PYTHONPATH=/home/bagus/github/nkululeko:$PYTHONPATH python tests/test_eer.py
```

## Technical Notes

- EER requires binary classification (2 classes)
- Best used with models that output probabilities (SVM with `probability=True`, XGBoost, neural networks)
- The implementation finds the threshold where FAR = FRR by minimizing |FAR - FRR|
- Returns the average of FAR and FNR at the optimal threshold

## References

- TorchMetrics EER: https://lightning.ai/docs/torchmetrics/stable/classification/eer.html
- Issue #15: https://github.com/bagustris/nkululeko/issues/15

## Future Enhancements

Potential improvements:
- Support for multi-class EER (one-vs-rest)
- ROC curve plotting with EER threshold marked
- DET (Detection Error Tradeoff) curve visualization
