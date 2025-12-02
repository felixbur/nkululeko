# Feature Correlation Plots (regplot)

## Overview

This tutorial shows how to use the `regplot` feature (added in Version 1.1.0 via PR #316) to visualize correlations between pairs of continuous acoustic features with either categorical or continuous target variables. These plots help you understand relationships between features and reveal patterns useful for model building.

## What is regplot?

`regplot` creates scatter (regression) plots for two continuous features. Optionally, a third variable (categorical or continuous) drives point coloring (and bubble size for continuous targets). Typical uses:
- Explore feature correlations
- Inspect how features relate to target labels
- Detect multivariate patterns prior to modeling
- Support hypothesis formation for feature engineering

### Highlights
- Two or three element specifications: `[feat_x, feat_y]` or `[feat_x, feat_y, target]`
- Categorical targets: color-coded regression plots
- Continuous targets: bubble plots (size + color gradient)
- Automatic Pearson correlation coefficient (PCC)
- Optional mixed linear model statistics (`print_stats = True`)
- Graceful feature name suggestions on typos

## Configuration Syntax

Add to the `[EXPL]` section of your INI file:

```ini
regplot = [[feat_1, feat_2], [feat_1, feat_2, target], ...]
```

Meaning:
- `[feat_x, feat_y]`: Plot two features, color by default class label (`target` from `[DATA]`).
- `[feat_x, feat_y, target]`: Plot two features, color (and optionally size) by explicit `target` variable.

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

### Target Variable Types
**Categorical** (e.g. `emotion`, `gender`): distinct colors per category (seaborn regression styling).

**Continuous** (e.g. `age`, `duration`, MOS scores): bubble plot where:
- Bubble size scales the continuous target (range 5–50 px)
- Bubble color uses a continuous colormap (default: viridis)

## Example 1: Categorical Target

```ini
[EXP]
root = ./
name = results/exp_emodb_regplot
runs = 1

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb.csv
emodb.type = csv
emodb.absolute_path = False
emodb.split_strategy = speaker_split
target = emotion
labels = ['anger', 'boredom', 'disgust', 'fear']

[FEATS]
type = ['praat']
praat.feature_set = basic
scale = standard

[EXPL]
regplot = [['duration', 'meanF0Hz']]
print_stats = True

[MODEL]
type = xgb
```

Produces a regression plot of `duration` vs `meanF0Hz` colored by emotion with PCC and (if enabled) mixed model stats.

## Example 2: Continuous Target (Bubble Plot)

```ini
[EXP]
root = ./
name = results/exp_age_continuous
runs = 1

[DATA]
databases = ['custom']
custom = ./data/custom/dataset.csv
custom.type = csv
custom.absolute_path = False
custom.split_strategy = speaker_split
target = age

[FEATS]
type = ['praat']
praat.feature_set = extended
scale = standard

[EXPL]
regplot = [
    ['duration', 'meanF0Hz', 'age'],
    ['HNR', 'localJitter', 'age']
]
print_stats = True

[MODEL]
type = xgb
```

Creates bubble plots with size + color representing `age`.

## Example 3: Multiple Combinations

From `examples/examples/exp_emodb_explore_features.ini`:

```ini
[EXPL]
regplot = [
    ['duration', 'meanF0Hz'],
    ['duration', 'meanF0Hz', 'age'],
    ['HNR', 'stdevF0Hz'],
    ['localJitter', 'HNR', 'gender']
]
feature_distributions = all
max_feats = 5
scatter = ['pca']
print_stats = True
```

Generates four regression/bubble plots with mixed usage of targets.

## Statistical Output (`print_stats = True`)
1. **Pearson Correlation Coefficient (PCC)** for the x–y feature pair.
2. **Mixed Linear Model** (speaker random effects) summarizing fixed effects, interactions, and variance components.

### Sample Output
```
DEBUG: plots: saved regplot to .../images/regplot_duration-meanF0Hz-class_label.png
DEBUG: plots:                  Mixed Linear Model Regression Results
=======================================================================
Model:                   MixedLM      Dependent Variable:      duration
No. Observations:        60           Method:                  REML    
No. Groups:              10           Scale:                   0.5665  
Min. group size:         3            Log-Likelihood:          -83.6068
Max. group size:         11           Converged:               Yes     
Mean group size:         6.0                                           
-----------------------------------------------------------------------
                            Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
-----------------------------------------------------------------------
Intercept                    6.122    1.814  3.375 0.001   2.567  9.677
emotion[T.happy]            -5.186    3.389 -1.530 0.126 -11.827  1.456
emotion[T.neutral]          -5.139    2.049 -2.508 0.012  -9.155 -1.123
emotion[T.sad]              -2.858    2.161 -1.322 0.186  -7.094  1.378
meanF0Hz                    -0.018    0.009 -2.046 0.041  -0.035 -0.001
emotion[T.happy]:meanF0Hz    0.025    0.017  1.500 0.134  -0.008  0.057
emotion[T.neutral]:meanF0Hz  0.025    0.010  2.409 0.016   0.005  0.045
emotion[T.sad]:meanF0Hz      0.021    0.012  1.661 0.097  -0.004  0.045
Group Var                    0.045    0.097                            
=======================================================================
```

**Interpretation:**
- Intercept: baseline duration for reference emotion
- Main effects: emotion category shifts duration; F0 has a slight negative relation
- Interaction terms: show modulation of F0 effect by emotion
- Group Var: speaker-level variability
- P>|z|: significance (< 0.05 noteworthy)

## Output Files
Saved under `results/<experiment_name>/images/` with naming pattern:
```
regplot_<feat_x>-<feat_y>-<target>.png
```
Examples:
- `regplot_duration-meanF0Hz-class_label.png`
- `regplot_duration-meanF0Hz-age.png`
- `regplot_HNR-localJitter-gender.png`

## Complete Working Example
```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_explore_features.ini
```
List resulting images:
```bash
ls results/exp_emodb_explore/images/
```

## Advanced Usage
### Combine with Feature Importance
```ini
[EXPL]
feature_distributions = all
model = ['tree', 'xgb']
max_feats = 5
regplot = [['feature_1', 'feature_2']]
```

### Alongside Dimensionality Reduction
```ini
[EXPL]
scatter = ['pca', 'tsne', 'umap']
regplot = [['duration', 'meanF0Hz']]
```

## Common Feature Pairs
Prosodic: `duration` vs `meanF0Hz`, `duration` vs `stdevF0Hz`

Voice Quality: `HNR` vs `localJitter`, `HNR` vs `shimmer`

Spectral: `spectral_centroid` vs `spectral_bandwidth`, `mfcc_1` vs `mfcc_2`

## Tips
1. Select theoretically related features
2. Standardize features (`scale = standard`)
3. Use sufficient samples for continuous bubble plots
4. Enable `print_stats = True` for deeper analysis
5. Batch multiple combinations for broader insight

## Troubleshooting
### Missing Feature
```
KeyError: 'feature_name'
```
Check spelling or list available columns (enable column value printing). Suggestions appear in the error message.

### Cluttered Plot / Overlap
Filter samples or switch to continuous target bubble plots.

### No Statistics
Ensure:
```ini
[EXPL]
print_stats = True
```
And install statsmodels if absent:
```bash
pip install statsmodels
```

## Related Options
```ini
[EXPL]
feature_distributions = all
plot_features = ['feat_a']
scatter = ['pca', 'tsne']
print_stats = True
sample_selection = all
```

## Implementation Notes
- `nkululeko/plots.py` (regplot())
- `nkululeko/feat_extract/feats_analyser.py`
- `nkululeko/utils/util.py` (`scale_to_range`, `df_to_cont_dict`)
- `nkululeko/utils/stats.py` (stat tests)

## References
- PR #316
- Issue #315
- Seaborn regplot / pairplot docs

## Changelog (1.1.0)
Added regression & bubble plots, PCC display, mixed linear model stats, feature suggestion on error, flexible 2/3 variable specification.

## Further Reading
See `ini_file.md` and other tutorials under the Tutorials section for complementary visualization and exploration techniques.
