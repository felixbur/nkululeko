# Predicting Speaker ID

This tutorial shows how to predict speaker identities in your audio data using pyannote speaker diarization.

**Reference**: [Nkululeko: Predict Speaker ID](http://blog.syntheticspeech.de/2024/11/07/nkululeko-predict-speaker-idwith-nkululekohttps-github-com-felixbur-nkululeko-since-version-0-85-0-the-acoustic-features-for-the-test-and-the-train-aka-dev-set-are-exported-to-the-project-s/)

## Overview

Since version 0.93.0, Nkululeko interfaces with [pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization (as an alternative to [silero](https://github.com/snakers4/silero-vad)).

There are **two modules** for speaker identification:

| Module | Scope | Use Case |
|--------|-------|----------|
| **SEGMENT** | Per-file | Find speakers within each audio file (e.g., one long recording) |
| **PREDICT** | Whole database | Find speakers across all files in the database |

> ⚠️ **Performance Note**: Both methods are slow on CPU. Best run on GPU.

## Requirements

- **HuggingFace Token**: Required for pyannote models
  - Get yours at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **GPU recommended**: CPU processing is very slow (no progress bar)

## Method 1: SEGMENT Module

Use this when you want to find different speakers **within each file** (e.g., diarizing a long conversation).

```ini
[EXP]
root = ./examples/results/
name = exp_emodb_segment_speaker

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = random
target = emotion

[SEGMENT]
method = pyannote
segment_target = _segmented
sample_selection = all

[MODEL]
hf_token = <your_huggingface_token>
device = cuda  ; or gpu

[FEATS]
type = ['os']
scale = standard
```

### SEGMENT Options

| Option | Description |
|--------|-------------|
| `method` | `pyannote` (speaker diarization) or `silero` (VAD only) |
| `segment_target` | Suffix for output files (e.g., `_segmented`) |
| `sample_selection` | Which samples to process: `all`, `train`, or `test` |
| `min_length` | Minimum segment length in seconds (optional) |
| `max_length` | Maximum segment length in seconds (optional) |

### Output

The SEGMENT module produces:
- New segmented audio files with speaker labels
- A distribution plot of detected speakers in the `images/` folder

## Method 2: PREDICT Module

Use this when you want to identify speakers **across the entire database** (e.g., clustering utterances by speaker).

```ini
[EXP]
root = ./examples/results/
name = exp_emodb_predict_speaker

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = random
target = emotion

[FEATS]
type = ['os']
scale = standard

[PREDICT]
targets = ['speaker']
sample_selection = all

[MODEL]
type = xgb
device = cuda
```

### PREDICT Options

| Option | Description |
|--------|-------------|
| `targets` | What to predict: `['speaker']`, `['gender']`, `['age']`, etc. |
| `sample_selection` | Which samples: `all`, `train`, or `test` |

### Available Prediction Targets

The PREDICT module supports multiple targets:
- `speaker` - Speaker identity
- `gender` - Male/Female
- `age` - Age estimation
- `snr` - Signal-to-noise ratio
- `valence`, `arousal`, `dominance` - Emotional dimensions
- `pesq`, `mos` - Speech quality metrics
- `text` - Transcription (via Whisper)

## Running the Experiments

### With SEGMENT module:

```bash
python -m nkululeko.segment --config examples/exp_emodb_segment_speaker.ini
```

### With PREDICT module:

```bash
python -m nkululeko.predict --config examples/exp_emodb_predict_speaker.ini
```

## Example Files

- [`exp_emodb_segment_speaker.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_segment_speaker.ini): SEGMENT-based speaker diarization
- [`exp_emodb_predict_speaker.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_predict_speaker.ini): PREDICT-based speaker identification
- [`exp_androids_segment.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_androids_segment.ini): Silero VAD segmentation

## Tips

1. **Use GPU**: Set `device = cuda` in `[MODEL]` section for 10x+ speedup
2. **HuggingFace token**: Required for pyannote; accept the model license on HuggingFace first
3. **Silero alternative**: Use `method = silero` for faster VAD-only segmentation (no speaker ID)
4. **Long files**: Use `max_length` to split very long segments

## Related Tutorials

- [Text Processing](text_processing.md): Transcription and translation
- [Segmentation Module](segment.md): Voice activity detection basics
