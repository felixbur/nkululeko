# Recurrent Networks (LSTM) on Frame Sequences

This tutorial shows how to train an LSTM on frame-level features with
nkululeko, for classification (`[MODEL] type = lstm`) and regression
(`[MODEL] type = lstm_reg`).

## Overview

Most nkululeko models see one feature vector per utterance: the features
are aggregated over time (e.g. opensmile functionals) before the model
gets them. A recurrent network instead reads the utterance as a sequence
of frames and can model how the signal develops over time. The label is
unchanged: still one value per utterance.

The frames come from the normal feature extraction with
`[FEATS] level = frame` (see [Sequence data](sequence_data.md)), so they
are cached, can be scaled, and can just as well be imported from a CSV
file. Currently opensmile gives frame output: its low-level descriptors,
e.g. 25 values per 10 ms frame for eGeMAPSv02.

## Quick Start

### 1. Classification on EmoDB

Runnable example:
[`examples/exp_emodb_lstm.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_lstm.ini),
which uses four emotion classes of EmoDB:

```ini
[EXP]
root = ./examples/results/
name = lstm_emodb
runs = 1
epochs = 20
save = True
[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
labels = ['anger', 'happiness', 'sadness', 'neutral']
target = emotion
[FEATS]
type = ['os']
level = frame
scale = standard
[MODEL]
type = lstm
```

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_lstm.ini
```

Scaling the features (`scale = standard`) matters here: the low-level
descriptors have very different ranges.

### 2. Your own frame features

Frame features computed elsewhere can be imported from a CSV with the
columns `file`, `start`, `end`, `frame` (all in seconds, `frame` relative
to the segment start) plus one column per feature:

```ini
[FEATS]
type = ['import']
import_file = ['my_frames.csv']
scale = standard
[MODEL]
type = lstm
```

### 3. A bidirectional LSTM

There is no separate model type for a BLSTM; switch it on in `[RNN]`:

```ini
[MODEL]
type = lstm
[RNN]
bidirectional = True
hidden_size = 64
num_layers = 2
```

### 4. Regression

For a continuous target (e.g. age), use `lstm_reg` and set
`[EXP] type = regression`. Loss and evaluation measure work as for
`mlp_reg`:

```ini
[EXP]
type = regression
[DATA]
target = age
[FEATS]
type = ['os']
level = frame
scale = standard
[MODEL]
type = lstm_reg
loss = mse
measure = ccc
```

### 5. The same frames for other models

With the same `[FEATS]` settings, any other model gets the frames pooled
to one vector per utterance (`[FEATS] pooling`, default mean and
standard deviation), which makes it easy to compare the LSTM with e.g.
an SVM on the same frame features:

```ini
[MODEL]
type = svm
```

## Output

The usual experiment output: per-epoch test results (UAR for
classification, the `measure` for regression), the confusion matrix or
regression plot, and the probabilities CSV in `results/<exp>/results/`.
With `save = True` the network weights are stored per epoch, as for the
other neural models.

## Technical details

- **Variable length.** Each utterance keeps its own number of frames. A
  batch is padded to its longest sequence; the padding is skipped by the
  LSTM (packed sequences) and left out of the average below.
- **Pooling inside the network.** The network averages the recurrent
  outputs over the utterance's frames and feeds the average to the
  output layer. Unlike using only the last hidden state, this works the
  same for one- and two-directional LSTMs.
- **Shared settings.** Learning rate (default 0.001), optimizer,
  scheduler, loss, dropout (`drop`, applied between stacked layers) and
  device come from `[MODEL]`, as for the other neural models.
  `[MODEL] layers` and `activation` do not apply; the network size is set
  with `[RNN] hidden_size` and `num_layers`.
- **Naming.** The section is called `[RNN]` because its keys apply to any
  recurrent cell type; only the model types are called `lstm`/`lstm_reg`.
- **DANN and SAM.** Domain-adversarial training (`[MODEL] dann_columns`)
  and SAM (`[MODEL] sam`) work as for the other neural models; the DANN
  heads read the time-pooled recurrent outputs.
- **Limitations.** `[FEATS] balancing`, `predict_sample`, SHAP
  explanations and ONNX export are not supported for the LSTM models.

## Related

[Sequence data](sequence_data.md) (the frame format),
[`ini_file.md`](ini_file.md#rnn) (`[RNN]` key reference),
[`emotion_prediction.md`](emotion_prediction.md).
