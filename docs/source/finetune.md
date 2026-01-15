# Finetuning Transformer Models

This tutorial shows how to finetune pretrained transformer models (like wav2vec2, WavLM, HuBERT) for your specific classification or regression task.

**Reference**: [Nkululeko: How to finetune a transformer model](http://blog.syntheticspeech.de/2024/05/29/nkululeko-how-to-finetune-a-transformer-model/)

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
- Maximum audio duration: 8 seconds by default (rest is ignored)

## Choosing a Pretrained Model

The default model is [facebook/wav2vec2-large-robust-ft-swbd-300h](https://huggingface.co/facebook/wav2vec2-large-robust-ft-swbd-300h).

Specify a different model:

```ini
[MODEL]
type = finetune
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

## Training Parameters

Configure deep learning hyperparameters:

```ini
[MODEL]
type = finetune
pretrained_model = microsoft/wavlm-base
learning_rate = 0.0001
batch_size = 16
device = cuda:0
duration = 10.5
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrained_model` | wav2vec2-large-robust | HuggingFace model name |
| `learning_rate` | 0.00001 | Learning rate |
| `batch_size` | 8 | Batch size (reduce if OOM) |
| `device` | cuda | Device: `cuda`, `cuda:0`, `cpu` |
| `duration` | 8 | Max audio duration in seconds |

## Loss Functions

Loss functions are automatically selected:

- **Classification**: Weighted cross-entropy
- **Regression**: Concordance correlation coefficient (CCC)

## Publishing to HuggingFace

To publish your finetuned model to HuggingFace Hub:

```ini
[MODEL]
type = finetune
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
pretrained_model = microsoft/wavlm-base
batch_size = 4
device = cuda
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
2. **Duration**: Long audio files are truncated to `duration` seconds
3. **Epochs**: Start with 5-10 epochs; use early stopping with dev set
4. **Model size**: Use `base` models for limited GPU; `large` for best performance
5. **Learning rate**: Default is usually fine; reduce if training is unstable

## Related Tutorials

- [Train/Dev/Test Splits](traindevtest.md): Proper evaluation with early stopping
- [Comparing Runs](compare_runs.md): Compare finetuned vs embedding approaches
