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

The LSTM models compute the frame sequence themselves from the audio,
the same way `aasist` and `finetune` read the audio directly. You
therefore set `[FEATS] type = []`, and choose the frames in the `[RNN]`
section instead:

- **melspec** (default): a log mel spectrogram
- **opensmile_lld**: opensmile low-level descriptors (e.g. eGeMAPS: 25
  values per 10 ms frame)

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
type = []
[MODEL]
type = lstm
[RNN]
feature_source = melspec
```

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_lstm.ini
```

### 2. Use opensmile low-level descriptors instead

```ini
[RNN]
feature_source = opensmile_lld
opensmile_set = eGeMAPSv02
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

For a continuous target (e.g. age), use `lstm_reg`. Loss and evaluation
measure work as for `mlp_reg`:

```ini
[DATA]
target = age
[FEATS]
type = []
[MODEL]
type = lstm_reg
loss = mse
measure = ccc
[RNN]
feature_source = melspec
```

## Output

The usual experiment output: per-epoch test results (UAR for
classification, the `measure` for regression), the confusion matrix or
regression plot, and the probabilities CSV in `results/<exp>/results/`.
With `save = True` the network weights are stored per epoch, as for the
other neural models.

## Technical details

- **Fixed length.** Each clip is converted to mono, resampled to 16 kHz
  and brought to `max_len` samples (default 64600, about 4 s): shorter
  clips are repeated (tiled), longer ones are cut. Every clip therefore
  gets the same number of frames, so batches need no padding or masking.
  Increase `max_len` if your utterances are long and the end matters.
- **Pooling.** The network averages the recurrent outputs over all frames
  and feeds the average to the output layer. Unlike using only the last
  hidden state, this works the same for one- and two-directional LSTMs
  and does not depend on where a tiled clip happens to end.
- **Features on the fly.** Frames are computed per clip during training
  and evaluation and are not cached. With `feature_source = opensmile_lld`
  this is slower than with `melspec`; `[MODEL] n_jobs` sets the number of
  DataLoader workers that compute them in parallel.
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
- **Limitations.** `predict_sample`, SHAP explanations and ONNX export are
  not supported, as for the other models that read raw audio (`adm`,
  `aasist`).

## Related

[`ini_file.md`](ini_file.md#rnn) (`[RNN]` key reference),
[`emotion_prediction.md`](emotion_prediction.md),
[`finetune.md`](finetune.md) (another model type that reads the audio
directly).
