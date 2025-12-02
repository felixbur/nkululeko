# Nkululeko Predict Module

The Nkululeko Predict module provides automated prediction capabilities for various audio characteristics using pre-trained models. This module allows you to add predicted labels to your datasets without training new models.

## Overview

The predict module (`nkululeko.predict`) automatically predicts labels for audio samples using existing models and adds them to your dataframe. It supports prediction for various targets including speaker identity, gender, age, emotions, and audio quality metrics.

## Usage

### Command Line Interface

```bash
python3 -m nkululeko.predict --config CONFIG_FILE
```

**Arguments:**
- `--config`: Path to the configuration file (default: `exp.ini`)

### Configuration

The predict module is controlled through the `[PREDICT]` section in your configuration file:

```ini
[PREDICT]
targets = ['gender', 'age', 'emotion']  # List of targets to predict
split = all                             # Which split to predict: all, train, or test
```

**Configuration Parameters:**
- `targets`: List of prediction targets (see available targets below)
- `split`: Data split selection
  - `all`: Predict for both train and test sets (default)
  - `train`: Predict only for training set
  - `test`: Predict only for test set

## Available Autopredict Targets

The following autopredict targets are available, each implemented in corresponding `ap_*.py` files:

### Speaker and Identity
- **`speaker`** (`ap_sid.py`): Speaker identification prediction
  - Uses speaker identification models to predict speaker identity

### Demographic Attributes
- **`gender`** (`ap_gender.py`): Biological sex prediction
  - Uses audEERING's agender model for gender classification
  - Predicts: male, female

- **`age`** (`ap_age.py`): Age prediction
  - Predicts speaker age using pre-trained age estimation models

### Emotional Attributes
- **`emotion`** (`ap_emotion.py`): Emotion classification
  - Uses emotion2vec models for emotion prediction
  - Supports multi-class emotion recognition

- **`arousal`** (`ap_arousal.py`): Arousal level prediction
  - Predicts emotional arousal (activation level)
  - Continuous value prediction

- **`valence`** (`ap_valence.py`): Valence prediction
  - Predicts emotional valence (positive/negative)
  - Continuous value prediction

- **`dominance`** (`ap_dominance.py`): Dominance prediction
  - Predicts emotional dominance dimension
  - Continuous value prediction

### Audio Quality Metrics
- **`mos`** (`ap_mos.py`): Mean Opinion Score prediction
  - Predicts subjective audio quality (MOS)
  - Range typically 1-5

- **`pesq`** (`ap_pesq.py`): PESQ score prediction
  - Perceptual Evaluation of Speech Quality
  - Objective speech quality metric

- **`stoi`** (`ap_stoi.py`): STOI prediction
  - Short-Time Objective Intelligibility measure
  - Speech intelligibility metric

- **`sdr`** (`ap_sdr.py`): Signal-to-Distortion Ratio prediction
  - Measures signal quality relative to distortion

- **`snr`** (`ap_snr.py`): Signal-to-Noise Ratio prediction
  - Estimates background noise level relative to speech signal

## Example Configuration

```ini
[EXP]
root = ./experiments/
name = audio_prediction_experiment

[DATA]
databases = ['mydata']
mydata = ./data/audio_files.csv
mydata.type = csv
target = emotion

[PREDICT]
targets = ['gender', 'age', 'arousal', 'valence', 'mos']
split = all

[PLOT]
name = prediction_results
```

## Workflow

1. **Load Configuration**: Read experiment configuration from INI file
2. **Load Datasets**: Load audio data according to DATA section
3. **Split Data**: Create train/test splits as specified
4. **Feature Extraction**: Extract features required for each predictor
5. **Prediction**: Apply pre-trained models for each target
6. **Add Labels**: Add predicted labels as new columns to the dataframe
7. **Save Results**: Save the augmented dataset with predictions

## Output

The predict module:
- Adds predicted labels as new columns to your dataset (e.g., `gender_pred`, `age_pred`)
- Saves the augmented dataset as `{dataset_name}_predicted.csv`
- Preserves original data while adding prediction columns

## Implementation Details

### Predictor Architecture
Each autopredict target follows a consistent pattern:
```python
class TargetPredictor:
    def __init__(self, df):
        self.df = df
        self.util = Util("targetPredictor")
    
    def predict(self, split_selection):
        # Extract features using appropriate feature extractor
        # Apply pre-trained model
        # Return dataframe with predictions
```

### Feature Extraction
Predictors use the `FeatureExtractor` class to extract relevant features:
- Audio features (e.g., OpenSMILE, wav2vec2)
- Specialized features for specific tasks (e.g., agender for gender)
- Pre-computed embeddings (e.g., emotion2vec for emotions)

### Model Integration
- Uses pre-trained models from various sources
- audEERING models for demographic prediction
- emotion2vec for emotional attributes
- Specialized audio quality assessment tools

## Dependencies

Different predictors may require additional dependencies:
- **audEERING models**: `audonnx`
- **emotion2vec**: `funasr`
- **Audio quality metrics**: Various specialized libraries

## Use Cases

1. **Dataset Augmentation**: Add demographic or emotional labels to unlabeled audio
2. **Quality Assessment**: Evaluate audio quality metrics for large datasets
3. **Multi-label Datasets**: Create comprehensive labels for audio collections
4. **Preprocessing**: Prepare datasets with rich metadata for downstream tasks
5. **Analysis**: Understand characteristics of audio datasets

## Tips and Best Practices

1. **Target Selection**: Choose targets relevant to your analysis goals
2. **Split Strategy**: Use `split = all` for complete dataset labeling
3. **Performance**: Some predictors require significant computational resources
4. **Validation**: Consider validating predictions on known subsets
5. **Integration**: Use predicted labels as features in subsequent experiments

## Error Handling

The module includes robust error handling:
- Validates target specifications
- Handles missing or corrupted audio files
- Provides informative error messages
- Continues processing when individual predictions fail

This predict module enables efficient automated labeling of audio datasets, supporting rapid prototyping and comprehensive audio analysis workflows in Nkululeko.
