# Finetuning Transformer Models

This tutorial shows how to finetune pretrained transformer models (like wav2vec2, WavLM, HuBERT) for your specific classification or regression task.

**Reference**: [Nkululeko: How to finetune a transformer model](http://blog.syntheticspeech.de/2024/05/29/nkululeko-how-to-finetune-a-transformer-model/)

## Finetuning vs. Normal Model Training

In normal Nkululeko training, `[FEATS]` extracts a fixed numeric feature vector for each sample once (e.g. openSMILE, wav2vec2 embeddings), and `[MODEL]` (e.g. `svm`, `xgb`, `mlp`, `cnn`) fits a classifier or regressor on top of those static features - the feature extractor itself never changes. Finetuning instead treats the pretrained transformer as the model itself: its own weights are updated directly on the raw audio via backpropagation, jointly adjusting the acoustic representation and the prediction head for your specific task.

| | Normal training | Finetuning |
|---|---|---|
| `[FEATS]` | extracts a fixed feature vector (e.g. `['os']`, `['wav2vec2']`) | must be empty (`[]`) - the transformer replaces this step |
| What gets trained | a classical/lightweight model (SVM, XGB, MLP, CNN, …) on top of static features | the pretrained transformer's own weights (optionally with some layers frozen, see [`freeze_layers`](#partial-finetuning-freezing-layers)) |
| Feature extractor | frozen, computed once and cached | trainable end-to-end by default |
| Compute cost | low - features are cached vectors, cheap to train on | high - a full forward/backward pass through the transformer per batch, GPU strongly recommended |
| Typical use | small/medium datasets, quick iteration | large datasets, best achievable performance |

## Overview

Since version 0.85.0, Nkululeko supports finetuning transformer models with [HuggingFace](https://huggingface.co/docs/transformers/training).

**Finetuning** means training the entire pretrained transformer with your data labels, as opposed to only using the last layer as embeddings (which is what `type = ['wav2vec2']` does in `[FEATS]`).

## When to Finetune vs Use Embeddings

| Approach | When to Use |
|----------|-------------|
| **Embeddings** (`[FEATS] type = ['wav2vec2']`) | Small datasets, quick experiments, limited GPU |
| **Finetuning** (`[MODEL] type = finetune`) | Large datasets, best performance, GPU available |

## Basic Configuration

To finetune a transformer model:

```ini
[EXP]
root = ./examples/results/
name = wavlm_finetuned
epochs = 5

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = speaker_split
target = emotion

[FEATS]
; Features should be empty for finetuning
type = []

[MODEL]
type = finetune
```

### Key Points

- `[FEATS] type = []` - Must be empty because the transformer model has its own CNN layers for acoustic feature extraction
- `[MODEL] type = finetune` - Triggers finetuning mode
- `[FINETUNE]` - Holds all finetuning-specific settings (optional; every key below has a default)
- Maximum audio duration: 8 seconds by default (rest is ignored)

## Choosing a Pretrained Model

The default model is [facebook/wav2vec2-large-robust-ft-swbd-300h](https://huggingface.co/facebook/wav2vec2-large-robust-ft-swbd-300h).

Specify a different model:

```ini
[MODEL]
type = finetune

[FINETUNE]
pretrained_model = microsoft/wavlm-base
```

### Popular Pretrained Models

| Model | Description |
|-------|-------------|
| `facebook/wav2vec2-large-robust-ft-swbd-300h` | Default, robust to noise |
| `microsoft/wavlm-base` | Good for speech tasks |
| `microsoft/wavlm-large` | Larger, better performance |
| `facebook/hubert-base-ls960` | HuBERT base model |
| `facebook/wav2vec2-base-960h` | Smaller, faster |

Nkululeko automatically builds the correct backbone architecture (wav2vec2, WavLM, or HuBERT) and loads each checkpoint's own preprocessing settings (e.g. input normalization) based on `pretrained_model` - no extra configuration needed regardless of which family you pick.

## Training Parameters

Configure deep learning hyperparameters in the `[FINETUNE]` section:

```ini
[MODEL]
type = finetune

[FINETUNE]
pretrained_model = microsoft/wavlm-base
learning_rate = 0.0001
batch_size = 16
device = 0
max_duration = 10.5
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrained_model` | `facebook/wav2vec2-large-robust-ft-swbd-300h` | HuggingFace model name |
| `learning_rate` | 0.0001 | Learning rate |
| `batch_size` | 8 | Batch size (reduce if OOM) |
| `device` | autodetect | Device: GPU index (e.g. `0`, or `0,1`), `cuda:0` (the index is extracted), or `cpu` |
| `max_duration` | 8 | Max audio duration in seconds |
| `freeze_layers` | 0 | Number of pretrained encoder layers (from the input side) to keep frozen; `0` finetunes the whole backbone |
| `num_layers` | (empty) | Total number of encoder layers to build the model with, truncating the pretrained architecture; empty/unset uses the pretrained model's full depth |
| `drop` | 0.1 | Dropout applied in the classification/regression head |
| `push_to_hub` | False | Upload the finetuned model to HuggingFace Hub |
| `balancing` | none | Training-set balancing: `ros`, `smote`, or `adasyn` |
| `loss` | `cross` (classification) / `1-ccc` (regression) | Loss function |
| `class_weight` | False | Weight the loss by inverse class frequency (classification only) |
| `measure` | `ccc` (regression only; classification always uses UAR) | Evaluation metric used for early-stopping/best-checkpoint selection |

### Partial Finetuning (Freezing Layers)

By default, finetuning updates the entire pretrained backbone (except the CNN feature extractor, which is always frozen). To freeze the first N transformer encoder layers and only train the rest plus the head - faster and less prone to overfitting on small datasets - set `freeze_layers`:

```ini
[FINETUNE]
pretrained_model = microsoft/wavlm-base
freeze_layers = 6
```

Only supported for the standard HuggingFace wav2vec2/WavLM/HuBERT backends; it's ignored (with a warning) for `emotion2vec*` pretrained models.

### Reducing Model Depth

`freeze_layers` keeps every layer but stops training some of them; `num_layers` instead builds a *smaller* model by truncating the pretrained architecture to the first N encoder layers, dropping the rest entirely (fewer parameters, faster inference, smaller checkpoint):

```ini
[FINETUNE]
pretrained_model = microsoft/wavlm-large
num_layers = 6
```

Leave it unset to use the pretrained model's full depth (the default).

Nkululeko validates that `0 <= freeze_layers < num_layers <= <the pretrained model's layer count>` (using the pretrained depth wherever `num_layers` is left unset) and fails fast with a clear error if not - this catches configs that would either exceed the pretrained checkpoint's depth or freeze the entire resulting backbone, leaving nothing to train.

### Early Stopping

Set `[MODEL] patience` (shared with every other model type, not a `[FINETUNE]` key) to stop finetuning once the dev-set metric stops improving, instead of always running the full `epochs` count:

```ini
[MODEL]
type = finetune
patience = 3

[FINETUNE]
pretrained_model = facebook/wav2vec2-large-robust-ft-swbd-300h
```

`patience` is in epochs, matching other model types. Internally finetuning evaluates 5 times per epoch, so this is scaled to 5x that many evaluation calls before stopping - you don't need to account for that yourself.

## Loss Functions

Loss functions are automatically selected:

- **Classification**: Weighted cross-entropy
- **Regression**: Concordance correlation coefficient (CCC)

## Publishing to HuggingFace

To publish your finetuned model to HuggingFace Hub:

```ini
[MODEL]
type = finetune

[FINETUNE]
push_to_hub = True
```

Make sure you're logged in to HuggingFace CLI first:
```bash
huggingface-cli login
```

## Complete Example

```ini
[EXP]
root = ./examples/results/
name = wavlm_finetuned
runs = 1
epochs = 10
save = True

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = speaker_split
target = emotion
labels = ['anger', 'happiness', 'neutral', 'sadness']

[FEATS]
type = []

[MODEL]
type = finetune

[FINETUNE]
pretrained_model = microsoft/wavlm-base
batch_size = 4
device = 0
; push_to_hub = True
```

## Output

The finetuning process produces:
- Best model checkpoint in the project folder
- HuggingFace logs (readable with TensorBoard)
- Training metrics and evaluation results

### Viewing Training Progress

```bash
tensorboard --logdir examples/results/wavlm_finetuned/
```

## Example Files

- [`exp_emodb_finetune.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_emodb_finetune.ini): Finetune WavLM on emoDB

## Running the Experiment

```bash
python -m nkululeko.nkululeko --config examples/exp_emodb_finetune.ini
```

## Tips

1. **GPU Memory**: Reduce `batch_size` if you get out-of-memory errors
2. **Duration**: Long audio files are truncated to `max_duration` seconds
3. **Epochs**: Start with 5-10 epochs; use early stopping with dev set
4. **Model size**: Use `base` models for limited GPU; `large` for best performance
5. **Learning rate**: Default is usually fine; reduce if training is unstable
6. **Freezing**: If finetuning is unstable or overfits on a small dataset, try `freeze_layers` before reducing `learning_rate`

## Related Tutorials

- [Train/Dev/Test Splits](traindevtest.md): Proper evaluation with early stopping
- [Comparing Runs](compare_runs.md): Compare finetuned vs embedding approaches
