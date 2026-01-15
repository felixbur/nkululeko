# Feature Correlation Plots (regplot)

## Overview

The `regplot` feature (added in Version 1.1.0 via PR #316) visualizes correlations between pairs of continuous acoustic features and optional targets. Use it to see how features relate to each other and to classification or regression targets, spot redundancy, and guide feature engineering before modeling.

### What regplot does
- Two- or three-element specs: `[feat_x, feat_y]` or `[feat_x, feat_y, target]`
- Categorical targets: color-coded regression plots
- Continuous targets: bubble plots (size + color gradient)
- Pearson correlation coefficient (PCC) overlay
- Optional mixed linear model statistics with `print_stats = True`
- Graceful feature name suggestions on typos

## Configuration

Add to the `[EXPL]` section of your INI file:

```ini
regplot = [[feat_1, feat_2], [feat_1, feat_2, target], ...]
```

| Format | Description |
|--------|-------------|
| `[feat1, feat2]` | Plot `feat1` vs `feat2`, colored by default target |
| `[feat1, feat2, target]` | Plot `feat1` vs `feat2`, colored (and for continuous targets also sized) by `target` |

### Target types
- **Categorical** (emotion, gender): per-class colors and regression lines
- **Continuous** (age, duration): bubble plots with color + size for the target value

### Examples
```ini
[EXPL]
regplot = [['duration', 'meanF0Hz']]
regplot = [['duration', 'meanF0Hz'], ['duration', 'stdevF0Hz']]
regplot = [['duration', 'meanF0Hz', 'emotion']]
regplot = [
    ['duration', 'meanF0Hz'],
    ['duration', 'meanF0Hz', 'age'],
    ['HNR', 'localJitter', 'gender']
]
```

## Example configuration (exp_emodb_explore_features.ini)

```ini
[EXP]
root = ./examples/results/
name = exp_emodb_explore
runs = 1
epochs = 1
save = True

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = random
labels = ['angry', 'happy', 'neutral', 'sad']
emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
target = emotion

[FEATS]
type = ['praat']
features = ['duration', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter']

[MODEL]
type = xgb

[EXPL]
sample_selection = all
feature_distributions = all
model = ['tree', 'xgb']
max_feats = 5
# Regplot: investigate feature correlations
regplot = [['duration', 'meanF0Hz'], ['duration', 'meanF0Hz', 'age']]
scatter = ['pca']
print_stats = True
```

Run it:

```bash
python -m nkululeko.explore --config examples/exp_emodb_explore_features.ini
```

## Interpreting the plots

### Categorical target (classification)
- X-axis: first feature (e.g., `duration`)
- Y-axis: second feature (e.g., `meanF0Hz`)
- Colors: target classes (e.g., angry, happy, neutral, sad)
- Regression lines: per-class trend lines

Example:
```ini
regplot = [['duration', 'meanF0Hz']]
```

![Regplot with Categorical Target](images/regplot_categorical.png)

### Continuous target (regression)
- X-axis: first feature
- Y-axis: second feature
- Colors: binned target values (grouped into ranges)
- Bubble size: represents target magnitude
- Regression lines: per-group trend lines

Example:
```ini
regplot = [['duration', 'meanF0Hz', 'age']]
```

![Regplot with Continuous Target](images/regplot_continuous.png)

## Output files

Plots are saved under `results/<experiment_name>/images/` using:

```
regplot_<feat_x>-<feat_y>-<target>.png
```

Examples:
- `regplot_duration-meanF0Hz-class_label.png`
- `regplot_duration-meanF0Hz-age.png`
- `regplot_HNR-localJitter-gender.png`

## Advanced usage

### Multiple regplots in one run
```ini
[EXPL]
regplot = [
    ['lld_mfcc3_sma3_median', 'lld_mfcc1_sma3_median'],
    ['lld_mfcc3_sma3_median', 'lld_F2frequency_sma3nz_median', 'age'],
    ['meanF0Hz', 'stdevF0Hz'],
    ['HNR', 'localJitter', 'gender']
]
```

### Using OpenSMILE features
```ini
[FEATS]
type = ['os']
set = eGeMAPSv02

[EXPL]
regplot = [['F0semitoneFrom27.5Hz_sma3nz_amean', 'jitterLocal_sma3nz_amean'],
           ['shimmerLocaldB_sma3nz_amean', 'HNRdBACF_sma3nz_amean']]
```

### Combine with other exploration options
```ini
[EXPL]
feature_distributions = all
scatter = ['pca', 'tsne', 'umap']
model = ['tree', 'xgb']
max_feats = 10
regplot = [['duration', 'meanF0Hz']]
print_stats = True
```

### Statistical output (`print_stats = True`)
1. PCC for the x–y feature pair
2. Mixed linear model (speaker random effects) summarizing fixed effects, interactions, and variance components

Sample excerpt:
```
DEBUG: plots: saved regplot to .../images/regplot_duration-meanF0Hz-class_label.png
DEBUG: plots:                  Mixed Linear Model Regression Results
...
emotion[T.neutral]:meanF0Hz  0.025    0.010  2.409 0.016   0.005  0.045
```

## Use cases
1. Feature selection and redundancy checks
2. Class separability visualization
3. Outlier and data quality inspection
4. Research insight into acoustic correlates

## Tips
1. Start with meaningful feature pairs; lean on domain knowledge
2. Standardize features (`scale = standard`) before plotting
3. For continuous targets, ensure enough samples for bubble plots
4. Enable `print_stats = True` when you want PCC and mixed-model stats
5. Batch several pairs to compare patterns quickly

## Troubleshooting

**Missing feature (`KeyError`)**: check spelling; the error suggests similar column names.

**Cluttered plots**: downsample, filter, or use a continuous target bubble plot.

**No statistics printed**: set `print_stats = True` and install statsmodels if needed:
```bash
pip install statsmodels
```

## Related features
- Feature distributions: `feature_distributions = all`
- Scatter plots: `scatter = ['pca', 'tsne', 'umap']`
- Feature importance: `model = ['tree', 'xgb']`, `max_feats = N`

## Implementation notes
- `nkululeko/plots.py` (`regplot`)
- `nkululeko/feat_extract/feats_analyser.py`
- `nkululeko/utils/util.py` (`scale_to_range`, `df_to_cont_dict`)
- `nkululeko/utils/stats.py` (stat tests)

## References and further reading
- PR #316, Issue #315
- Seaborn regplot / pairplot docs
- Blog: How to investigate correlations of specific features (Dec 2025)
- Blog: How to plot distributions of feature values (Feb 2023)
- See `ini_file.md` and other tutorials for complementary visualization techniques
