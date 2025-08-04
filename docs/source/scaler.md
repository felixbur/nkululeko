# Feature Scaling in Nkululeko

Feature scaling is a crucial preprocessing step in machine learning that standardizes the range of features to improve model performance and convergence. The Nkululeko framework provides a comprehensive `Scaler` class that offers multiple scaling strategies to normalize speech features.

## Table of Contents

- [Overview](#overview)
- [Available Scaling Methods](#available-scaling-methods)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The `Scaler` class in Nkululeko (`nkululeko/scaler.py`) handles feature normalization across training, development, and test sets. It ensures that:

- Features are scaled consistently across all datasets
- The scaling parameters are learned only from the training set
- Different scaling strategies can be applied based on data characteristics
- Speaker-specific normalization is supported

## Available Scaling Methods

### 1. Standard Scaling (`standard`)
**Z-score normalization** - transforms features to have zero mean and unit variance.

```ini
[FEATS]
scale = standard
```

**Formula:** `(x - mean) / std`

**Use case:** Most commonly used method, works well when features follow a normal distribution.

### 2. Robust Scaling (`robust`)
**Robust to outliers** - uses median and interquartile range instead of mean and standard deviation.

```ini
[FEATS]
scale = robust
```

**Formula:** `(x - median) / IQR`

**Use case:** Recommended when features contain outliers that could skew standard scaling.

### 3. Min-Max Scaling (`minmax`)
**Range normalization** - scales features to a fixed range [0, 1].

```ini
[FEATS]
scale = minmax
```

**Formula:** `(x - min) / (max - min)`

**Use case:** When you need features bounded to a specific range, especially for neural networks.

### 4. Max-Abs Scaling (`maxabs`)
**Absolute maximum scaling** - scales by the maximum absolute value.

```ini
[FEATS]
scale = maxabs
```

**Formula:** `x / max(|x|)`

**Use case:** Preserves sparsity in sparse datasets and handles both positive and negative values.

### 5. Normalizer (`normalizer`)
**L2 normalization** - scales individual samples to have unit norm.

```ini
[FEATS]
scale = normalizer
```

**Use case:** When the direction of the data vector is more important than the magnitude.

### 6. Power Transformer (`powertransformer`)
**Gaussian-like transformation** - applies power transformations to make data more Gaussian.

```ini
[FEATS]
scale = powertransformer
```

**Use case:** When features have skewed distributions and you want to make them more normal.

### 7. Quantile Transformer (`quantiletransformer`)
**Uniform/Gaussian mapping** - maps features to uniform or Gaussian distribution.

```ini
[FEATS]
scale = quantiletransformer
```

**Use case:** When you want to reduce the impact of outliers and enforce a specific distribution.

### 8. Binning (`bins`)
**Categorical binning** - converts continuous features into three categorical bins.

```ini
[FEATS]
scale = bins
```

**Output:** Features are converted to strings: "0" (low), "0.5" (medium), "1" (high)
**Thresholds:** 33rd and 66th percentiles of the training data

**Use case:** When you want to discretize continuous features for tree-based models or categorical analysis.

### 9. Speaker-wise Scaling (`speaker`)
**Per-speaker normalization** - applies standard scaling individually for each speaker.

```ini
[FEATS]
scale = speaker
```

**Use case:** When speaker-specific characteristics should be normalized, useful for speaker-independent emotion recognition.

## Configuration

### Basic Configuration

Add the scaling configuration to the `[FEATS]` section of your INI file:

```ini
[FEATS]
type = ['os']  # Feature type
set = eGeMAPSv02  # Feature set
scale = standard  # Scaling method
```

### Advanced Configuration Examples

#### Robust scaling with OpenSMILE features:
```ini
[FEATS]
type = ['os']
set = ComParE_2016
level = functionals
scale = robust
```

#### Min-max scaling for neural networks:
```ini
[FEATS]
type = ['spectra']
scale = minmax

[MODEL]
type = cnn
```

#### Speaker-wise normalization:
```ini
[FEATS]
type = ['os']
scale = speaker

[DATA]
# Ensure speaker information is available
target = emotion
```

#### Binning for tree-based models:
```ini
[FEATS]
type = ['os']
scale = bins

[MODEL]
type = xgb
```

## Usage Examples

### Complete Experiment Configuration

```ini
[EXP]
root = ./experiments/
name = emotion_recognition_robust_scaling
type = classification

[DATA]
databases = ['emodb']
emodb = /path/to/emodb
target = emotion
labels = ['anger', 'happiness', 'neutral', 'sadness']

[FEATS]
type = ['os']
set = eGeMAPSv02
level = functionals
scale = robust  # Using robust scaling for outlier resistance

[MODEL]
type = svm
C_val = 1.0
kernel = rbf
```

### Comparing Different Scaling Methods

You can compare different scaling methods by running separate experiments:

**Experiment 1: Standard scaling**
```ini
[EXP]
name = emotion_standard_scaling
[FEATS]
scale = standard
```

**Experiment 2: Robust scaling**
```ini
[EXP]
name = emotion_robust_scaling
[FEATS]
scale = robust
```

**Experiment 3: Min-max scaling**
```ini
[EXP]
name = emotion_minmax_scaling
[FEATS]
scale = minmax
```

## Best Practices

### 1. Choosing the Right Scaling Method

| Data Characteristics | Recommended Scaler | Reason |
|---------------------|-------------------|--------|
| Normal distribution, few outliers | `standard` | Classical z-score normalization |
| Contains outliers | `robust` | Uses median/IQR, less sensitive to outliers |
| Need bounded range [0,1] | `minmax` | Explicit range control |
| Sparse data | `maxabs` | Preserves sparsity |
| Skewed distributions | `powertransformer` | Makes data more Gaussian |
| Many outliers | `quantiletransformer` | Robust distribution mapping |
| Tree-based models | `bins` | Can improve interpretability |
| Speaker variability | `speaker` | Normalizes per-speaker differences |

### 2. Neural Network Considerations

For neural networks, consider:
- `minmax` for bounded inputs
- `standard` for well-behaved distributions
- Avoid `bins` as neural networks work better with continuous features

### 3. SVM Considerations

SVMs benefit from scaled features:
- `standard` or `robust` are typically good choices
- `minmax` ensures all features contribute equally

### 4. Tree-based Model Considerations

Tree-based models (XGBoost, Random Forest) are generally scale-invariant:
- Scaling may not be necessary
- `bins` can improve interpretability
- Standard scaling doesn't hurt and may help with some implementations

### 5. Cross-database Experiments

When working with multiple databases:
- Ensure consistent scaling across databases
- `robust` or `quantiletransformer` may be more stable across different recording conditions

## API Reference

### Scaler Class

```python
class Scaler:
    """Class to normalize speech features."""
    
    def __init__(self, train_data_df, test_data_df, train_feats, test_feats, 
                 scaler_type, dev_x=None, dev_y=None):
        """
        Initialize the scaler.
        
        Parameters:
        -----------
        train_data_df : pd.DataFrame
            Training dataframe with speaker information (needed for speaker scaling)
        test_data_df : pd.DataFrame  
            Test dataframe with speaker information
        train_feats : pd.DataFrame
            Training features dataframe
        test_feats : pd.DataFrame
            Test features dataframe
        scaler_type : str
            Type of scaling: 'standard', 'robust', 'minmax', 'maxabs', 
            'normalizer', 'powertransformer', 'quantiletransformer', 'bins', 'speaker'
        dev_x : pd.DataFrame, optional
            Development data dataframe
        dev_y : pd.DataFrame, optional
            Development features dataframe
        """
    
    def scale(self):
        """
        Scale features based on the configured scaling method.
        
        Returns:
        --------
        tuple
            (train_scaled, test_scaled) or (train_scaled, dev_scaled, test_scaled)
        """
    
    def scale_all(self):
        """Scale all datasets using the configured scaler."""
        
    def speaker_scale(self):
        """Apply speaker-wise scaling."""
        
    def bin_to_three(self):
        """Convert features to three bins: low, medium, high."""
```

### Key Methods

#### `scale()`
Main method that applies the selected scaling strategy.

#### `scale_all()`
Handles scaling for non-speaker-specific methods.

#### `speaker_scale()`
Applies scaling per speaker for speaker-wise normalization.

#### `bin_to_three()`
Implements the binning strategy, converting continuous features to categorical bins.

### Return Values

The scaler returns scaled DataFrames in the same format as the input:
- Same indices as input features
- Same column names as input features
- Scaled/transformed values according to the selected method

For the `bins` method, values are returned as strings: "0", "0.5", "1".

## Error Handling

The scaler includes robust error handling:

```python
# Invalid scaler type
scaler = Scaler(..., scaler_type="invalid")
# Raises: ValueError with message about unknown scaler

# Missing speaker information for speaker scaling
# Will raise appropriate error if speaker column is missing
```

## Integration with Nkululeko Pipeline

The scaler is automatically integrated into the Nkululeko pipeline:

1. Features are extracted according to `[FEATS]` configuration
2. Scaler is applied if `scale` parameter is specified
3. Scaled features are passed to the model for training/testing

No manual intervention is required - just specify the scaling method in your INI file.

---

For more information about feature extraction and model configuration, see:
- [Feature Extraction Documentation](nkululeko.feat_extract.rst)
- [INI File Configuration Reference](ini_file.md)
- [Model Documentation](nkululeko.models.rst)
