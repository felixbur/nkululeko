# Bundle: Export and Inference

Nkululeko provides a **bundle** module to export a trained experiment as a self-contained, portable directory. This bundle can then be shared with others and used for inference on new audio files without requiring the original training data or configuration.

## Overview

A bundle directory contains everything needed for inference:

- `manifest.json` — metadata and artifact paths
- `inference.ini` — sanitized config (no training paths or private data)
- `model.pkl` — the trained model
- `scaler.pkl` — the fitted scaler (if scaling was used)
- `label_encoder.pkl` — the label encoder (for classification)
- `feature_schema.json` — feature column names and dimensions
- `README.md` — human-readable description

## Tutorial: Polish Emotion Recognition Bundle

This tutorial walks through training a model on the Polish emotion recognition dataset, exporting it as a bundle, and using the bundle for inference on new audio files.

### Prerequisites

1. Install nkululeko (see [Installation](installation.rst)).
2. Download and prepare the Polish dataset (see `data/polish/README.md`):

```bash
cd data/polish
unzip polish_speech_emotions.zip
python3 process_database.py
cd ../..
```

### Step 1: Train the Model

First, train a model using the Polish dataset. Use the tutorial config file `tutorials/tut_bundle_polish.ini`:

```ini
[EXP]
root = ./
name = results/exp_polish_bundle
runs = 1
epochs = 1
save = True

[DATA]
databases = ['train', 'dev', 'test']
train = ./data/polish/polish_train.csv
train.type = csv
train.absolute_path = False
train.split_strategy = train
dev = ./data/polish/polish_dev.csv
dev.type = csv
dev.absolute_path = False
dev.split_strategy = train
test = ./data/polish/polish_test.csv
test.type = csv
test.absolute_path = False
test.split_strategy = test
target = emotion
labels = ['anger', 'neutral', 'fear']

[FEATS]
type = ['os']
scale = standard

[MODEL]
type = xgb
save = True
```

Key settings:
- `[EXP] save = True` — saves the experiment state so it can be exported.
- `[MODEL] save = True` — saves the trained model file.
- `[FEATS] scale = standard` — applies standard scaling (bundled with the export).

Run the training:

```bash
python3 -m nkululeko.nkululeko --config tutorials/tut_bundle_polish.ini
```

### Step 2: Export the Bundle

After training, export the model as a bundle:

```bash
python3 -m nkululeko.bundle --config tutorials/tut_bundle_polish.ini
```

By default, the bundle is saved to `<root>/<name>/export/`. You can specify a custom output directory:

```bash
python3 -m nkululeko.bundle --config tutorials/tut_bundle_polish.ini --output ./my_polish_bundle
```

The exported bundle directory will contain:

```
my_polish_bundle/
├── manifest.json
├── inference.ini
├── model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── feature_schema.json
└── README.md
```

### Step 3: Run Inference with the Bundle

Use the `nkululeko.infer` module to predict emotions on new audio files using the exported bundle.

#### Predict a single file

```bash
python3 -m nkululeko.infer ./my_polish_bundle --file /path/to/audio.wav
```

#### Predict multiple files

```bash
python3 -m nkululeko.infer ./my_polish_bundle --file file1.wav file2.wav file3.wav
```

#### Predict all audio files in a folder

```bash
python3 -m nkululeko.infer ./my_polish_bundle --folder /path/to/audio_folder --outfile predictions.csv
```

#### Predict files from a CSV list

```bash
python3 -m nkululeko.infer ./my_polish_bundle --list files.csv --outfile predictions.csv
```

### Step 4: Interpret Results

For classification tasks, the output includes:
- Per-class probabilities (e.g., `anger`, `neutral`, `fear`)
- The `predicted` column with the most likely label

Example output for a single file:

```
/path/to/audio.wav    anger: 0.85
/path/to/audio.wav    neutral: 0.10
/path/to/audio.wav    fear: 0.05
/path/to/audio.wav    predicted: anger
```

When using `--outfile`, results are saved as a CSV file with an audformat segmented index.

## Programmatic API

You can also use the bundle for inference in Python scripts:

```python
from nkululeko.infer import infer_from_bundle

results = infer_from_bundle(
    bundle_dir="./my_polish_bundle",
    files=["audio1.wav", "audio2.wav"],
)
print(results)
```

## Notes

- The bundle is self-contained and does not require the original training data or configuration.
- Feature extraction is performed automatically during inference using the same settings used during training.
- The bundle can be shared across machines as long as the same Python and nkululeko versions are available (check `manifest.json` for version info).
- For the Polish dataset, the labels are: `anger`, `neutral`, `fear`.
- Set `[EXP] save = True` and `[MODEL] save = True` in your training config before exporting.
