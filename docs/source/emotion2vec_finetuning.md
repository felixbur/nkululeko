# Emotion2vec Finetuning in Nkululeko

This document describes how to use emotion2vec models for finetuning in nkululeko.

## Overview

Emotion2vec finetuning allows you to fine-tune pre-trained emotion2vec models on your specific audio classification or regression tasks, similar to how wav2vec2, WavLM, and HuBERT models can be fine-tuned.

## Configuration

To use emotion2vec finetuning, set the following in your INI configuration file:

```ini
[MODEL]
type = finetune
pretrained_model = emotion2vec-base  # or emotion2vec, emotion2vec-large, emotion2vec-seed
learning_rate = 0.00001
batch_size = 8
epochs = 10
```

## Supported Models

The following emotion2vec model variants are supported:

- `emotion2vec` - Maps to `emotion2vec/emotion2vec_base` (768 dimensions)
- `emotion2vec-base` - Maps to `emotion2vec/emotion2vec_base` (768 dimensions)
- `emotion2vec-large` - Maps to `emotion2vec/emotion2vec_plus_large` (1024 dimensions)
- `emotion2vec-seed` - Maps to `emotion2vec/emotion2vec_plus_seed` (768 dimensions)

You can also specify the full model path directly:
```ini
pretrained_model = emotion2vec/emotion2vec_base
```

## Requirements

Emotion2vec finetuning requires the following additional dependencies:

```bash
pip install funasr
```

Note: Models are loaded from HuggingFace Hub (not ModelScope).

## Example Configuration

Here's a complete example configuration for emotion2vec finetuning:

```ini
[EXP]
root = ./experiments/
name = emotion2vec_finetune_experiment
epochs = 10
runs = 3

[DATA]
databases = ['emodb']
emodb = ./data/emodb/
emodb.split_strategy = train_split
target = emotion
labels = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']

[FEATS]
type = os
set = eGeMAPSv02

[MODEL]
type = finetune
pretrained_model = emotion2vec-base
learning_rate = 0.00001
batch_size = 4
epochs = 10
drop = 0.1

[PLOT]
name = emotion2vec_results
```

## Technical Details

The emotion2vec finetuning implementation:

1. Uses FunASR's AutoModel for the backbone emotion2vec model (from HuggingFace Hub)
2. Adds a classification/regression head on top of the embeddings (768 dimensions for base/seed models, 1024 for large models)
3. Integrates with HuggingFace's Trainer API for training
4. Supports both classification and regression tasks
5. Handles audio preprocessing through temporary file conversion for FunASR compatibility

## Limitations

- Audio processing involves temporary file creation for FunASR compatibility, which may be slower than direct tensor processing
- Requires FunASR dependency (but uses HuggingFace Hub for model downloads)
- Currently supports the same audio preprocessing pipeline as other transformer models (16kHz sampling rate)
- Large models require more GPU memory due to 1024-dimensional embeddings
