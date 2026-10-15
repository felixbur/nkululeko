# Sequence Data: Frames Inside Segments

This page describes how nkululeko represents data that varies over time
inside a sample: frame-level features (e.g. opensmile low-level
descriptors) and frame-level labels (e.g. continuous arousal ratings).

## Overview

A sample in nkululeko is a **segment** of an audio file, identified by
`file`, `start` and `end`. Usually it has one feature vector and one
label.

Sequence data adds one more level inside the segment, the **frame**:

| level | meaning |
|---|---|
| `file` | audio file |
| `start`, `end` | the segment within the file (the sample) |
| `frame` | start time of the frame, **relative to the segment start** |

Each row is one frame; the columns are features (or labels). The first
frame of every segment has `frame = 0`. Because frames carry a time, two
tables with different frame rates (e.g. features every 10 ms, ratings
every 40 ms) can be matched to each other.

The usual one-row-per-segment data is the special case without frames
and stays exactly as before.

## CSV layout

The same layout is used for features and labels. All times are in
seconds:

```
file,start,end,frame,F0,loudness
wav/03a01Fa.wav,0.0,1.9,0.00,31.2,0.41
wav/03a01Fa.wav,0.0,1.9,0.01,31.0,0.44
wav/03a01Fa.wav,0.0,1.9,0.02,30.7,0.47
wav/03a01Nc.wav,0.0,2.3,0.00,28.9,0.12
...
```

## Frame-level features

### Extracting them

Set `[FEATS] level = frame` (`lld` also works). Currently opensmile
supports it, giving its low-level descriptors (25 per frame for
eGeMAPSv02, one frame every 10 ms):

```ini
[FEATS]
type = ['os']
level = frame
```

Other feature types ignore the setting (with a warning) and give one
vector per sample.

### Importing them

Frame features computed elsewhere can be imported in the CSV layout
above; nkululeko recognizes them by the `frame` column:

```ini
[FEATS]
type = ['import']
import_file = ['my_frames.csv']
```

### Which model gets what

- **Sequence models** (`lstm`, `lstm_reg`) get the frames; each sequence
  keeps its own length. See [RNN classification](rnn_classification.md).
- **All other models** (svm, xgb, mlp, ...) get one vector per sample:
  the frames are pooled with the statistics in `[FEATS] pooling`
  (default mean and standard deviation). So frame features work with
  every model:

```ini
[FEATS]
type = ['os']
level = frame
pooling = ['mean', 'std', 'max']
[MODEL]
type = svm
```

Each feature then gets one column per statistic, e.g.
`Loudness_sma3_mean`, `Loudness_sma3_std`, `Loudness_sma3_max`.

Pooling also happens for `predict_sample`, bundles and the feature
analysis in `explore`, so these work as before.

### What else to know

- Feature scaling (`[FEATS] scale`) works on frames: the scaler is fitted
  on all training frames (`scale = speaker`: all frames of a speaker).
- `[FEATS] balancing` (SMOTE etc.) does not work with frame sequences,
  as it would create single frames instead of whole sequences.
- Several feature types can be combined for a sequence model only if
  they have the same frames.
- Frame-level features are cached separately from the functionals, and
  frame-level runs have `_level-frame` in their name.

## Frame-level labels

Labels that change over time (e.g. continuous arousal ratings as in
AVEC or RECOLA) use the same CSV layout, with label columns instead of
feature columns:

```
file,start,end,frame,arousal,valence
rec1.wav,0.0,60.0,0.00,0.12,-0.05
rec1.wav,0.0,60.0,0.04,0.14,-0.04
...
```

Labels and features often have different frame rates. The helper
`align_frame_labels` gives each feature frame the latest label at or
before it, within the same segment:

```python
from nkululeko.utils.sequences import align_frame_labels, read_sequence_csv

labels = read_sequence_csv("ratings.csv")    # e.g. every 40 ms
feats = read_sequence_csv("my_frames.csv")   # e.g. every 10 ms
aligned = align_frame_labels(labels, feats)  # one label row per feature frame
```

Experiments still use one label per segment; training and evaluating on
frame-level labels (sequence-to-sequence models) is not supported yet.

## Related

[`ini_file.md`](ini_file.md#feats) (`FEATS.level`, `FEATS.pooling`,
`FEATS.import_file`), [RNN classification](rnn_classification.md).
