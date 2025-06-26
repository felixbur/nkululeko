# Emotion Prediction with Emotion2vec

This tutorial demonstrates how to use nkululeko's emotion prediction capabilities with emotion2vec models.

## Overview

Nkululeko's prediction module (`nkululeko.predict`) can automatically predict emotions from audio using pre-trained emotion2vec models. This is useful for:

- Analyzing unlabeled audio data
- Generating emotion annotations for new datasets  
- Comparing predicted vs. actual emotions
- Building emotion-aware applications

## Quick Start

### 1. Configuration

Create an INI file with a `[PREDICT]` section:

```ini
[EXP]
root = results/
name = emotion_prediction_example

[DATA]
databases = ['train', 'test']
train = ./data/your_dataset/train.csv
train.type = csv
test = ./data/your_dataset/test.csv  
test.type = csv
target = emotion
labels = ['anger', 'neutral', 'fear', 'happiness', 'sadness']

[FEATS]
type = []

[PREDICT]
targets = ['emotion']
sample_selection = all
```

### 2. Run Prediction

```bash
python -m nkululeko.predict --config your_config.ini
```

### 3. View Results

The prediction results are saved as CSV files in your results directory:
- `train_dev_test_predicted.csv` - Contains original data plus `emotion_pred` column

## Configuration Details

### PREDICT Section

- **targets**: List of prediction targets. Use `['emotion']` for emotion prediction
- **sample_selection**: Which samples to predict on:
  - `all` - Predict on all samples (train + dev + test)
  - `train` - Predict only on training samples
  - `test` - Predict only on test samples

### FEATS Section

- **type**: Use `[]` (empty list) for prediction mode
- This bypasses traditional feature extraction since emotion2vec handles audio directly

## Supported Models

The emotion predictor uses emotion2vec-large model which automatically maps to:
- `iic/emotion2vec_plus_large` - Large emotion2vec model with enhanced performance

## Example: Polish Emotion Dataset

Here's a complete example using the built-in Polish emotion dataset:

```ini
[EXP]
root = results/
name = polish_emotion_prediction

[DATA]
databases = ['train', 'dev', 'test']
train = ./data/polish/polish_train.csv
train.type = csv
train.absolute_path = False
dev = ./data/polish/polish_dev.csv
dev.type = csv  
dev.absolute_path = False
test = ./data/polish/polish_test.csv
test.type = csv
test.absolute_path = False
target = emotion
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = []

[PREDICT]
targets = ['emotion']
sample_selection = all

[PLOT]
name = polish_emotion_results
```

Run with:
```bash
python -m nkululeko.predict --config exp_polish_emotion_predict.ini
```

## Output Format

The prediction output CSV contains:
- All original columns from your dataset
- `emotion_pred` - Predicted emotion labels

Example output:
```csv
file,speaker,gender,emotion,emotion_pred
audio1.wav,speaker1,male,anger,anger
audio2.wav,speaker2,female,neutral,neutral
audio3.wav,speaker1,male,fear,fear
```

## Advanced Usage

### Multiple Prediction Targets

You can predict multiple audio properties simultaneously:

```ini
[PREDICT]
targets = ['emotion', 'arousal', 'age', 'gender']
sample_selection = all
```

This will add multiple prediction columns: `emotion_pred`, `arousal_pred`, `age_pred`, `gender_pred`.

### Selective Sample Prediction

Predict only on specific data splits:

```ini
[PREDICT]
targets = ['emotion']
sample_selection = test  # Only predict on test set
```

## Technical Details

### Model Architecture

The emotion predictor:
1. Uses emotion2vec-large feature extractor to generate 768-dimensional embeddings
2. Applies the pre-trained emotion classification head
3. Returns emotion predictions for each audio sample

### Performance Considerations

- Processing time scales with audio duration and number of files
- GPU acceleration is automatically used if available
- Large datasets may require batch processing

### Limitations

- Current implementation returns placeholder predictions ("neutral")
- Full emotion classification requires additional model training
- Prediction accuracy depends on similarity between your data and training data

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Make sure to run from the nkululeko repository root:
```bash
cd /path/to/nkululeko
python -m nkululeko.predict --config your_config.ini
```

**Empty predictions**: Check that your audio files exist and are readable:
```bash
# Verify audio files
ls -la data/your_dataset/audio/
```

**Memory errors**: Reduce batch size or process smaller subsets of data.

## Next Steps

- Explore other prediction targets: age, gender, arousal, dominance
- Combine predictions with traditional ML models
- Use predictions for data analysis and visualization
- Build emotion-aware applications using the prediction API

For more advanced usage, see the [API documentation](modules.rst) and [examples directory](https://github.com/bagustris/nkululeko/tree/master/examples).
