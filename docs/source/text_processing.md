# Text Processing: Transcribe, Translate, and Classify

This tutorial demonstrates how to use Nkululeko's text processing pipeline to:

1. **Transcribe** audio to text using Whisper speech-to-text
2. **Translate** text between languages using Google Translate
3. **Classify** text topics using zero-shot classification

This is useful when you want to analyze the linguistic content of speech
databases, especially for cross-lingual analysis.

## Overview

The pipeline consists of three steps, each invoking the unified
[`nkululeko.predict`](predict.md) module with a different autopredict target:

```
Audio → [--model text]              → Text
Text  → [--model translation]       → English Text
Text  → [--model textclassification] → Topic Labels
```

The output CSV of one step is fed into the `--list` argument of the next.

## Prerequisites

- Nkululeko >= 1.6.0
- A speech database (we use Berlin EmoDB as an example)
- Required dependencies: `openai-whisper`, `googletrans`

## Step 1: Transcribe Audio to Text

Use Whisper (via `transformers`) to transcribe audio to text.

### Configuration (`exp_emodb_predict_text.ini`)

The config carries only the source-language setting; the input / output / model
choice is on the command line.

```ini
[EXP]
root = ./examples/results
name = exp_emodb_predict_text
language = de
```

### Run transcription

```bash
python -m nkululeko.predict \
    --list ./data/emodb/emodb_files.csv \
    --model text \
    --config examples/exp_emodb_predict_text.ini \
    --outfile ./emodb_transcribed.csv
```

### Output

The output CSV preserves the original columns and adds a `text` column:

```csv
file,start,end,emotion,text
./data/emodb/wav/03a01Fa.wav,0 days,,happiness,Der Lappen liegt auf dem Eisschrank.
./data/emodb/wav/03a01Nc.wav,0 days,,neutral,Der Lappen liegt auf dem Eisschrank.
```

## Step 2: Translate Text to English

Translate the German transcriptions to English using Google Translate.

### Configuration (`exp_emodb_translate.ini`)

```ini
[EXP]
root = ./examples/results
name = exp_emodb_translate
language = de
target_language = en
```

### Run translation

The input is the CSV produced in step 1 (it already contains the `text`
column expected by the translation predictor):

```bash
python -m nkululeko.predict \
    --list ./emodb_transcribed.csv \
    --model translation \
    --config examples/exp_emodb_translate.ini \
    --language es \
    --outfile ./emodb_translated.csv
```

> **Note**: `--language es` overrides both `EXP.language` and
> `PREDICT.target_language` from the INI. For `--model translation` only the
> target language matters, so the output column is named after `--language`
> (`es` here). Drop `--language` to fall back to the INI's `target_language`.

### Output

```csv
file,start,end,emotion,text,es
./data/emodb/wav/03a01Fa.wav,0 days,,happiness,Der Lappen liegt auf dem Eisschrank.,El trapo está sobre la nevera.
```

## Step 3: Classify text topics

Zero-shot classification with a multilingual XLM-RoBERTa model.

### Configuration (`exp_emodb_textclassifier.ini`)

```ini
[EXP]
root = ./examples/results
name = emodb_textclassifier

[FEATS]
textclassifier.candidates = ["sadness", "anger", "neutral", "happiness", "fear", "disgust", "boredom"]
```

### Run classification

```bash
python -m nkululeko.predict \
    --list ./emodb_translated.csv \
    --model textclassification \
    --config examples/exp_emodb_textclassifier.ini \
    --outfile ./emodb_classified.csv
```

### Zero-shot classification

The text classifier uses
[`joeddav/xlm-roberta-large-xnli`](https://huggingface.co/joeddav/xlm-roberta-large-xnli),
a zero-shot model that can classify text into **any categories you define**
without further training. Customize the candidates for your use case:

```ini
# Sentiment analysis
textclassifier.candidates = ["positive", "negative", "neutral"]

# Topic classification
textclassifier.candidates = ["sports", "politics", "technology", "entertainment"]

# Intent detection
textclassifier.candidates = ["question", "statement", "command", "greeting"]
```

### Output

```csv
file,classification_winner,sadness,anger,neutral,happiness,fear,disgust,boredom
./data/emodb/wav/03a01Fa.wav,neutral,0.116,0.141,0.359,0.059,0.089,0.121,0.114
```

## Complete pipeline

Run all three steps in sequence, piping each output to the next input:

```bash
python -m nkululeko.predict --list ./data/emodb/emodb_files.csv  --model text               --config examples/exp_emodb_predict_text.ini    --outfile transcribed.csv
python -m nkululeko.predict --list transcribed.csv               --model translation        --config examples/exp_emodb_translate.ini       --outfile translated.csv
python -m nkululeko.predict --list translated.csv                --model textclassification --config examples/exp_emodb_textclassifier.ini  --outfile classified.csv
```

## Troubleshooting

### `KeyError: 'text'`

The translation step needs a `text` column in the input CSV. Make sure step 1
finished successfully and that you pass its output to step 2 via `--list`.

### Slow transcription

Whisper is slow on CPU. Use GPU if available:

```ini
[MODEL]
device = cuda
```

### Google Translate rate limits

For large datasets you may hit translation API rate limits. Consider:

- Splitting the list into smaller chunks
- Adding delays between requests
- Switching to an alternative translation service

## Use cases

1. **Cross-lingual emotion analysis**: analyse emotional content in
   non-English speech.
2. **Content analysis**: extract topics and themes from speech recordings.
3. **Dataset enrichment**: add linguistic features to audio datasets.
4. **Multilingual research**: compare linguistic patterns across languages.

## Related tutorials

- [Predict module](predict.md): full documentation of the unified prediction module
  and its autopredict targets.
- [Hello World](hello_world_aud.md): getting started with Nkululeko.
- [Explore module](explore.md): visualize and analyse your data.

## References

- [Whisper](https://github.com/openai/whisper): OpenAI's speech recognition model
- [XLM-RoBERTa-XNLI](https://huggingface.co/joeddav/xlm-roberta-large-xnli): zero-shot classification model
- [Blog: How to add textual transcriptions](https://blog.syntheticspeech.de/2025/06/26/nkululeko-how-to-add-textual-transcriptions-to-your-data/)
- [Blog: How to translate transcriptions](http://blog.syntheticspeech.de/2025/07/14/nkululelo-how-to-translate-your-textual-transcriptions/)
- [Blog: How to predict topics](http://blog.syntheticspeech.de/2025/10/16/nkululeko-how-to-predict-topics-for-your-texts/)
