# Emotion Prediction with Emotion2vec

This tutorial demonstrates how to use Nkululeko's emotion prediction
capabilities with `emotion2vec` models.

## Overview

The unified prediction module ([`nkululeko.predict`](predict.md)) can
automatically predict emotions from audio. With `--model emotion` the
`emotion` autopredict target is used, which is currently backed by
`emotion2vec`. This is useful for:

- Analyzing unlabeled audio data
- Generating emotion annotations for new datasets
- Comparing predicted vs. actual emotions
- Building emotion-aware applications

## Quick Start

### 1. Predict emotion for a few files

```bash
python -m nkululeko.predict --file sample1.wav sample2.wav --model emotion
```

This writes `sample1_result.txt` and `sample2_result.txt` next to each input
file (and prints the same predictions to stdout).

### 2. Predict for a whole folder

```bash
python -m nkululeko.predict \
    --folder ./recordings \
    --model emotion \
    --outfile ./recordings_emotion.csv
```

### 3. Augment an existing CSV with an `emotion_pred` column

```bash
python -m nkululeko.predict \
    --list ./your_dataset.csv \
    --model emotion \
    --outfile ./your_dataset_with_emotion.csv
```

All original columns of `your_dataset.csv` are preserved. A new `emotion_pred`
column is appended.

## Output format

Example output CSV after `--list ... --model emotion`:

```csv
file,start,end,speaker,gender,emotion,emotion_pred
audio1.wav,0 days,,speaker1,male,anger,anger
audio2.wav,0 days,,speaker2,female,neutral,neutral
audio3.wav,0 days,,speaker1,male,fear,fear
```

The audformat segmented index (`file`, `start`, `end`) is preserved when the
input CSV is a valid audformat file. For a plain CSV, the first column is
interpreted as the audio path.

## Combining with other autopredict targets

Each invocation of `nkululeko.predict` runs one prediction target. To enrich
your CSV with multiple targets (emotion + arousal + age + gender), run the
command multiple times, threading the output of one run into the input of the
next:

```bash
python -m nkululeko.predict --list data.csv --model emotion --outfile step1.csv
python -m nkululeko.predict --list step1.csv --model arousal --outfile step2.csv
python -m nkululeko.predict --list step2.csv --model age --outfile step3.csv
python -m nkululeko.predict --list step3.csv --model gender --outfile final.csv
```

`final.csv` will contain the original columns plus `emotion_pred`,
`arousal_pred`, `age_pred` and `gender_pred`.

## Supported models

The emotion predictor uses an `emotion2vec` feature extractor, currently
`iic/emotion2vec_plus_large`.

## Technical details

### Pipeline

1. `emotion2vec-large` extracts 768-dimensional embeddings for each segment.
2. The pretrained emotion head produces the predicted emotion label.
3. The predicted labels are written as a new `emotion_pred` column.

### Performance

- Processing time scales with audio duration and the number of files.
- GPU acceleration is used automatically when available.
- Large datasets may require running in batches by splitting the input CSV.

### Limitations

- The bundled `EmotionPredictor` currently emits a placeholder label
  (`neutral`) until the emotion head is fully wired up.
- Prediction quality depends on how closely your data resembles the
  emotion2vec training distribution.

## Troubleshooting

**ModuleNotFoundError**: Run from a directory where `nkululeko` is installed
or available on `PYTHONPATH`:

```bash
python -m nkululeko.predict --file your.wav --model emotion
```

**Empty predictions**: Make sure the audio files exist and are readable:

```bash
ls -la ./recordings/
```

**Memory errors**: Process smaller subsets of data or run on a machine with
more RAM/VRAM.

## Next steps

- Try other autopredict targets: `age`, `gender`, `arousal`, `dominance`,
  `mos`, `snr`, `speaker`. See [predict.md](predict.md).
- Use the predict module in `--type model` mode to apply a Nkululeko model
  you trained yourself: see [demo.md](demo.md).
- Combine predicted labels with traditional ML pipelines via
  [`nkululeko.nkululeko`](usage.md).
