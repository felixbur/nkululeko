# How to Align Databases

This tutorial explains how to combine and align multiple databases that have different label schemes for related tasks. This is useful when you want to leverage data from one domain (e.g., emotion) to improve prediction in another related domain (e.g., stress).

## Overview

Sometimes you want to combine databases that are similar but don't label exactly the same phenomena. For example:
- You have limited **stress** data but many **emotion** databases
- You want to use **angry** samples as **stressed** and **happy/neutral** as **non-stressed**

Nkululeko provides several configuration options to align databases:
- **Column renaming** (`colnames`)
- **Label mapping** (`mapping`)
- **Sample filtering** (`filter`)
- **Target table selection** (`target_tables`)

## Configuration Options

### Column Renaming: `colnames`

Rename columns to align with your target task:

```ini
emodb.colnames = {"emotion": "stress"}
```

This renames the `emotion` column to `stress`.

### Label Mapping: `mapping`

Map original labels to new categories:

```ini
emodb.mapping = {"anger": "stress", "disgust": "stress", "neutral": "no stress", "sadness": "no stress"}
```

### Sample Filtering: `filter`

Select only specific samples based on column values:

```ini
# Keep only anger, neutral, and happiness samples
emodb.filter = [["stress", ["anger", "neutral", "happiness"]]]
```

### Target Tables: `target_tables`

Specify which tables contain the target labels:

```ini
emodb.target_tables = ["emotion"]
```

## Example: Emotion to Stress Mapping

This example shows how to convert Berlin EmoDB emotion labels into binary stress labels.

### Configuration: `exp_emodb_stress.ini`

```ini
[EXP]
root = ./examples/results/
name = emodb_stress
save_test = ./examples/results/emodb_stress/test.csv
epochs = 5

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
# Specify where target values come from
emodb.target_tables = ["emotion"]
# Rename emotion column to stress
emodb.colnames = {"emotion": "stress"}
# Keep only these emotion categories
emodb.filter = [["stress", ["anger", "neutral", "sadness", "disgust"]]]
# Map emotions to stress labels
emodb.mapping = {"anger": "stress", "disgust": "stress", "neutral": "no stress", "sadness": "no stress"}
emodb.split_strategy = speaker_split
# Define final labels
labels = ["stress", "no stress"]
target = stress

[FEATS]
type = ['os']

[MODEL]
type = mlp
layers = [64, 12]
drop = [.3, .4]

[PLOT]
uncertainty_threshold = 0.3
```

### Run the Experiment

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_stress.ini
```

## Advanced: Combining Multiple Databases

You can combine databases with different label schemes by aligning them to a common target.

### Example: EmoDB + SUSAS for Stress Detection

```ini
[DATA]
databases = ['emodb', 'susas']

# EmoDB configuration
emodb = ./data/emodb/emodb
emodb.target_tables = ["emotion"]
emodb.colnames = {"emotion": "stress"}
emodb.filter = [["stress", ["anger", "neutral", "happiness"]]]
emodb.mapping = {"anger": "stress", "neutral": "no stress", "happiness": "no stress"}
# Use all emodb for training
emodb.split_strategy = train

# SUSAS configuration
susas = ./data/susas/
# Map ternary stress labels to binary
susas.mapping = {'0,1': 'no stress', '2': 'stress'}
susas.split_strategy = speaker_split

target = stress
labels = ["stress", "no stress"]
```

### Key Points

1. **EmoDB is used only for training** (`split_strategy = train`)
2. **SUSAS is split into train/test** (`split_strategy = speaker_split`)
3. **Both databases use the same target labels** (`stress`, `no stress`)

## Multi-Database Alignment with Root Files

For complex multi-database setups, use a separate configuration file for database roots:

### Root Configuration: `data_roots.ini`

```ini
[DATA]
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}

crema-d = ./data/crema-d/crema-d/1.3.0/
crema-d.split_strategy = specified
crema-d.colnames = {'sex':'gender'}
crema-d.target_tables = ['emotion.categories.desired.test','emotion.categories.desired.train']
crema-d.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
```

### Main Configuration

```ini
[EXP]
root = ./examples/results/multidb
databases = ['emodb', 'crema-d']

[DATA]
root_folders = ./examples/data_roots.ini
target = emotion
labels = ['angry', 'happy', 'sad', 'neutral']

[FEATS]
type = ['os']
scale = standard

[MODEL]
type = mlp
```

## Configuration Reference

| Option | Description | Example |
|--------|-------------|---------|
| `colnames` | Rename columns | `{"emotion": "stress"}` |
| `mapping` | Map label values | `{"anger": "stress", "neutral": "no stress"}` |
| `filter` | Filter samples by column values | `[["column", ["val1", "val2"]]]` |
| `target_tables` | Tables containing target labels | `["emotion"]` |
| `split_strategy` | How to split data | `train`, `test`, `speaker_split`, `random` |

## Use Cases

1. **Cross-domain transfer**: Use emotion data for stress detection
2. **Label harmonization**: Combine databases with different label schemes
3. **Data augmentation**: Add out-of-domain data to training
4. **Multi-corpus experiments**: Train on multiple databases with aligned labels

## Tips

- **In-domain data usually works better**: Adding out-of-domain data doesn't always help
- **Use a third database for evaluation**: When combining databases, evaluate on held-out data
- **Check label distributions**: Ensure balanced classes after mapping
- **Document your mappings**: Keep track of how labels were aligned

## Related Tutorials

- [Multi-Database Experiments](multidb.md)
- [Data Balancing](balance.md)
- [INI File Reference](ini_file.md)

## Reference

- [Blog: How to align databases](http://blog.syntheticspeech.de/2025/08/06/nkululeko-how-to-align-databases/)
