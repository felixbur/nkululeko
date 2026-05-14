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

Use this when you want to identify speakers **across a list of audio files**
(e.g., clustering utterances by speaker). The unified
[`nkululeko.predict`](predict.md) module dispatches `--model speaker` to the
`speaker` autopredict target.

```bash
python -m nkululeko.predict \
    --list ./data/emodb/emodb_files.csv \
    --model speaker \
    --outfile ./emodb_speakers.csv
```

The output CSV preserves the original columns of `emodb_files.csv` and adds a
`speaker_pred` column. The same command works with `--folder` or `--file`.

### Available autopredict targets

`--model` accepts any of the autopredict targets, including:

- `speaker` — speaker identity
- `gender` — male/female
- `age` — age estimation
- `snr` — signal-to-noise ratio
- `valence`, `arousal`, `dominance` — emotional dimensions
- `pesq`, `mos`, `sdr`, `stoi` — speech quality metrics
- `emotion` — emotion classification
- `text`, `translation`, `textclassification` — text processing

See [predict.md](predict.md) for the full list and per-target details.

## Running the Experiments

### With the SEGMENT module:

```bash
python -m nkululeko.segment --config examples/exp_emodb_segment_speaker.ini
```

### With the PREDICT module:

```bash
python -m nkululeko.predict \
    --list ./data/emodb/emodb_files.csv \
    --model speaker \
    --outfile ./emodb_speakers.csv
```

## Example Files

- [`exp_emodb_segment_speaker.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_segment_speaker.ini): SEGMENT-based speaker diarization
- [`exp_androids_segment.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_androids_segment.ini): Silero VAD segmentation

## Tips

1. **Use GPU**: Set `device = cuda` in `[MODEL]` section for 10x+ speedup
2. **HuggingFace token**: Required for pyannote; accept the model license on HuggingFace first
3. **Silero alternative**: Use `method = silero` for faster VAD-only segmentation (no speaker ID)
4. **Long files**: Use `max_length` to split very long segments

## Related Tutorials

- [Text Processing](text_processing.md): Transcription and translation
- [Segmentation Module](segment.md): Voice activity detection basics
