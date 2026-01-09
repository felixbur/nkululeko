# Linguistic Features with BERT

This tutorial shows how to use BERT embeddings to model linguistic (semantic) content of speech, either alone or combined with acoustic features.

**Reference**: [Nkululeko: How to explicitly model linguistics](http://blog.syntheticspeech.de/2025/07/22/nkululeko-how-to-explicitly-model-linguistics/)

## Overview

Speech emotion recognition typically relies on acoustic features like pitch, energy, and spectral characteristics. However, **what** is being said (the linguistic content) can be just as important as **how** it is said.

Nkululeko supports BERT (Bidirectional Encoder Representations from Transformers) embeddings to capture the semantic meaning of transcribed speech. This is particularly useful when:

- You have transcripts available in your dataset
- The spoken content is relevant to your classification task
- You want to combine linguistic and acoustic information

## Requirements

Your dataset must have a `text` column containing transcripts. If your column has a different name (e.g., "Utterance", "transcript"), use the `colnames` option to rename it.

## Basic BERT Features

To use only BERT linguistic features:

```ini
[EXP]
root = ./
name = exp_meld_bert
; Set language for BERT model
language = en

[DATA]
databases = ['train', 'test']
train = ./data/meld/meld_train.csv
train.type = csv
train.split_strategy = train
test = ./data/meld/meld_test.csv
test.type = csv
test.split_strategy = test
; Rename column to 'text' if needed
colnames = {'Utterance': 'text'}
target = emotion
labels = ['anger', 'joy', 'neutral', 'sadness']

[FEATS]
type = ['bert']
scale = standard

[MODEL]
type = svm
```

## Combining BERT with Acoustic Features

To leverage both linguistic and acoustic information:

```ini
[FEATS]
; Combine BERT with OpenSMILE acoustic features
type = ['bert', 'os']
os.set = eGeMAPSv02
scale = standard
```

This creates a feature vector combining:
- **BERT embeddings** (768 dimensions from bert-base-uncased)
- **OpenSMILE features** (88 features from eGeMAPSv02)

## BERT Model Selection

By default, Nkululeko uses `bert-base-uncased`. You can specify a different model:

```ini
[FEATS]
type = ['bert']
; Use multilingual BERT
bert.model = bert-base-multilingual-cased
```

Common BERT models:
- `bert-base-uncased`: English, 110M parameters (default)
- `bert-base-cased`: English, case-sensitive
- `bert-base-multilingual-cased`: 104 languages
- `bert-large-uncased`: English, 340M parameters

## Language Setting

The `language` option in `[EXP]` helps select appropriate models:

```ini
[EXP]
; For German text
language = de

; For English text  
language = en
```

## Using with Transcription

If you don't have transcripts, you can first use Whisper to transcribe:

```ini
[DATA]
; First experiment: transcribe
[PREDICT]
targets = ['text']
```

Then use the generated `text` column for BERT features in a subsequent experiment.

## Example Files

- [`exp_meld_bert.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_meld_bert.ini): BERT-only features
- [`exp_meld_bert_os.ini`](https://github.com/felixbur/nkululeko/blob/main/examples/exp_meld_bert_os.ini): BERT + OpenSMILE combined

## Running the Experiment

```bash
python -m nkululeko.nkululeko --config examples/exp_meld_bert.ini
```

## Tips

1. **Memory**: BERT models require significant GPU memory. Use `device = cpu` if needed.
2. **Text quality**: BERT performance depends on transcript quality.
3. **Feature scaling**: Always use `scale = standard` when combining different feature types.
4. **Combining features**: Multi-modal (linguistic + acoustic) often outperforms single modality.

## Related Tutorials

- [Text Processing Pipeline](text_processing.md): Transcribe and translate speech
- [Feature Correlations](regplot.md): Explore feature importance
