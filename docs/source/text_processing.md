# Text Processing: Transcribe, Translate, and Classify

This tutorial demonstrates how to use Nkululeko's text processing pipeline to:
1. **Transcribe** audio to text using Whisper speech-to-text
2. **Translate** text between languages using Google Translate
3. **Classify** text topics using zero-shot classification

This is useful when you want to analyze the linguistic content of speech databases, especially for cross-lingual analysis.

## Overview

The pipeline consists of three steps, each using `nkululeko.predict` module:

```
Audio → [Transcribe] → Text → [Translate] → English Text → [Classify] → Topic Labels
```

## Prerequisites

- Nkululeko >= 1.0.1
- A speech database (we use Berlin EmoDB as an example)
- Required dependencies: `openai-whisper`, `googletrans`

## Step 1: Transcribe Audio to Text

First, we convert audio to text using OpenAI's Whisper model via the `targets = ['text']` prediction.

### Configuration: `exp_emodb_predict_text.ini`

```ini
# Example INI file for transcribing emodb audio to text using speech-to-text
# Tutorial: https://blog.syntheticspeech.de/2025/06/26/nkululeko-how-to-add-textual-transcriptions-to-your-data/
# This uses Whisper (via transformers) to transcribe German audio to text
# Run with: python -m nkululeko.predict --config examples/exp_emodb_predict_text.ini
# Output: A CSV file with 'text' column containing transcriptions

[EXP]
root = ./examples/results
name = exp_emodb_predict_text
# Language for Whisper transcription (German for emodb)
language = de

[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = random
target = emotion

[FEATS]
type = []

[MODEL]
type = svm

[PREDICT]
# Transcribe audio to text using speech-to-text (Whisper)
targets = ['text']
sample_selection = all
```

### Key Configuration Options

| Option | Description |
|--------|-------------|
| `language = de` | Source language for Whisper (ISO 639-1 code) |
| `targets = ['text']` | Enables speech-to-text transcription |
| `sample_selection = all` | Transcribe all samples in the database |

### Run Transcription

```bash
python -m nkululeko.predict --config examples/exp_emodb_predict_text.ini
```

### Output

The output CSV will include a new `text` column with transcriptions:

```csv
file,emotion,text
./data/emodb/wav/03a01Fa.wav,happiness,Der Lappen liegt auf dem Eisschrank.
./data/emodb/wav/03a01Nc.wav,neutral,Der Lappen liegt auf dem Eisschrank.
```

## Step 2: Translate Text to English

Next, we translate the German transcriptions to English using Google Translate.

### Configuration: `exp_emodb_translate.ini`

```ini
# Example INI file for translating German emodb transcriptions to English
# Tutorial: http://blog.syntheticspeech.de/2025/07/14/nkululelo-how-to-translate-your-textual-transcriptions/
# This uses Google Translate to translate text from German (de) to English (en)
# Prerequisites: 
#   1. Run exp_emodb_predict_text.ini first to get text transcriptions
#   2. Or have a dataset with a 'text' column
# Run with: python -m nkululeko.predict --config examples/exp_emodb_translate.ini

[EXP]
root = ./examples/results
name = exp_emodb_translate
# Source language of the text (German for emodb)
language = de

[DATA]
databases = ['emodb']
# Use the output from exp_emodb_predict_text.ini which contains the 'text' column
emodb = ./examples/results/exp_emodb_predict_text/results/all_predicted.csv
emodb.type = csv
emodb.split_strategy = random
target = emotion

[FEATS]
type = []

[MODEL]
type = svm

[PREDICT]
targets = ['translation']
# Target language for translation (default: en)
target_language = en
sample_selection = all
```

### Key Configuration Options

| Option | Description |
|--------|-------------|
| `language = de` | Source language (in `[EXP]` section) |
| `targets = ['translation']` | Enables text translation |
| `target_language = en` | Target language (default: English) |
| `emodb.type = csv` | Input is a CSV file (from Step 1) |

### Important Notes

- The input CSV must have a `text` column (from Step 1)
- If you get a `KeyError: 'text'` error, delete the cache: `rm -rf ./examples/results/exp_emodb_translate/store/`

### Run Translation

```bash
python -m nkululeko.predict --config examples/exp_emodb_translate.ini
```

### Output

The output CSV will include a new `translation` column:

```csv
file,emotion,text,translation
./data/emodb/wav/03a01Fa.wav,happiness,Der Lappen liegt auf dem Eisschrank.,The rag is on the freezer.
```

## Step 3: Classify Text Topics

Finally, we classify the translated text using zero-shot classification with a multilingual XLM-RoBERTa model.

### Configuration: `exp_emodb_textclassifier.ini`

```ini
[EXP]
root = ./examples/results
name = emodb_textclassifier

[DATA]
databases = ['emodb']
emodb = ./examples/results/exp_emodb_translate/results/all_predicted.csv
emodb.type = csv
emodb.split_strategy = random
labels = ['anger', 'happiness']
target = emotion

[FEATS]
type = ['os']
store_format = csv

[MODEL]
type = svm

[PREDICT]
targets = ['textclassification']
textclassifier.candidates = ["sadness", "anger", "neutral", "happiness", "fear", "disgust", "boredom"]

[PLOT]
uncertainty_threshold = .4
```

### Key Configuration Options

| Option | Description |
|--------|-------------|
| `targets = ['textclassification']` | Enables zero-shot text classification |
| `textclassifier.candidates` | List of category labels to predict |

### Zero-Shot Classification

The text classifier uses [joeddav/xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli), a zero-shot model that can classify text into **any categories you define** without training.

You can customize the candidates for your use case:

```ini
# For sentiment analysis
textclassifier.candidates = ["positive", "negative", "neutral"]

# For topic classification
textclassifier.candidates = ["sports", "politics", "technology", "entertainment"]

# For intent detection
textclassifier.candidates = ["question", "statement", "command", "greeting"]
```

### Run Classification

```bash
python -m nkululeko.predict --config examples/exp_emodb_textclassifier.ini
```

### Output

The output includes:
- `classification_winner`: The predicted category
- Individual columns with logits for each candidate class

```csv
file,classification_winner,sadness,anger,neutral,happiness,fear,disgust,boredom
./data/emodb/wav/03a01Fa.wav,neutral,0.116,0.141,0.359,0.059,0.089,0.121,0.114
```

## Complete Pipeline

Run all three steps in sequence:

```bash
# Step 1: Transcribe
python -m nkululeko.predict --config examples/exp_emodb_predict_text.ini

# Step 2: Translate (German → English)
python -m nkululeko.predict --config examples/exp_emodb_translate.ini

# Step 3: Classify topics
python -m nkululeko.predict --config examples/exp_emodb_textclassifier.ini
```

## Troubleshooting

### KeyError: 'text'
The translation step requires a `text` column. Ensure:
1. Step 1 (transcription) completed successfully
2. Delete cached files: `rm -rf ./examples/results/exp_emodb_translate/store/`

### Slow transcription
Whisper transcription can be slow on CPU. Use GPU if available:
```ini
[MODEL]
device = cuda
```

### Google Translate rate limits
For large datasets, you may hit rate limits. Consider:
- Adding delays between requests
- Using batch processing
- Using alternative translation services

## Use Cases

1. **Cross-lingual emotion analysis**: Analyze emotional content in non-English speech
2. **Content analysis**: Extract topics and themes from speech recordings
3. **Dataset enrichment**: Add linguistic features to audio datasets
4. **Multilingual research**: Compare linguistic patterns across languages

## Related Tutorials

- [Predict Module](predict.md): Full documentation of autopredict targets
- [Hello World](hello_world_aud.md): Getting started with Nkululeko
- [Explore Module](explore.md): Visualize and analyze your data

## References

- [Whisper](https://github.com/openai/whisper): OpenAI's speech recognition model
- [XLM-RoBERTa-XNLI](https://huggingface.co/joeddav/xlm-roberta-large-xnli): Zero-shot classification model
- [Blog: How to add textual transcriptions](https://blog.syntheticspeech.de/2025/06/26/nkululeko-how-to-add-textual-transcriptions-to-your-data/)
- [Blog: How to translate transcriptions](http://blog.syntheticspeech.de/2025/07/14/nkululelo-how-to-translate-your-textual-transcriptions/)
- [Blog: How to predict topics](http://blog.syntheticspeech.de/2025/10/16/nkululeko-how-to-predict-topics-for-your-texts/)
