# Pre-processing script for MELD-ST dataset

This directory contains the pre-processing script and data files for the MELD-ST (Multimodal EmotionLines Dataset - Speech Translation) dataset.

## Dataset Description

MELD-ST is a multimodal speech translation dataset that extends the original MELD dataset with parallel audio files in multiple languages. It contains emotion and sentiment annotations for dialogues from the TV show "Friends".

The dataset includes:
- **ENG_DEU**: English-German parallel audio with emotion/sentiment labels
- **ENG_JPN**: English-Japanese parallel audio with emotion/sentiment labels

### Emotions
7 emotion classes: anger, disgust, fear, joy, neutral, sadness, surprise

### Sentiments
3 sentiment classes: negative, neutral, positive

## Dataset Statistics

### ENG_DEU Language Pair
#### English (ENG)
- Total samples: 11,642
- Speakers: 291 unique
- Train: 9,314 samples
- Valid: 1,164 samples  
- Test: 1,164 samples

#### German (DEU)
- Total samples: 11,642
- Speakers: 291 unique
- Train: 9,314 samples
- Valid: 1,164 samples
- Test: 1,164 samples

### ENG_JPN Language Pair
#### English (ENG)
- Total samples: 10,085
- Speakers: 281 unique
- Train: 8,069 samples
- Valid: 1,008 samples
- Test: 1,008 samples

#### Japanese (JPN)
- Total samples: 10,085
- Speakers: 281 unique
- Train: 8,069 samples
- Valid: 1,008 samples
- Test: 1,008 samples

## Generated Files

The preprocessing script generates the following CSV files in nkululeko-compatible format:

### Language-Specific Split Files

#### ENG_DEU Dataset
- `meld_st_eng_deu_eng_train.csv` - English training split for ENG_DEU
- `meld_st_eng_deu_eng_valid.csv` - English validation split for ENG_DEU  
- `meld_st_eng_deu_eng_test.csv` - English test split for ENG_DEU
- `meld_st_eng_deu_deu_train.csv` - German training split for ENG_DEU
- `meld_st_eng_deu_deu_valid.csv` - German validation split for ENG_DEU
- `meld_st_eng_deu_deu_test.csv` - German test split for ENG_DEU

#### ENG_JPN Dataset
- `meld_st_eng_jpn_eng_train.csv` - English training split for ENG_JPN
- `meld_st_eng_jpn_eng_valid.csv` - English validation split for ENG_JPN
- `meld_st_eng_jpn_eng_test.csv` - English test split for ENG_JPN
- `meld_st_eng_jpn_jpn_train.csv` - Japanese training split for ENG_JPN
- `meld_st_eng_jpn_jpn_valid.csv` - Japanese validation split for ENG_JPN
- `meld_st_eng_jpn_jpn_test.csv` - Japanese test split for ENG_JPN

### Combined Files (All Splits)
- `meld_st_eng_deu_eng_all.csv` - All English splits combined for ENG_DEU
- `meld_st_eng_deu_deu_all.csv` - All German splits combined for ENG_DEU
- `meld_st_eng_jpn_eng_all.csv` - All English splits combined for ENG_JPN
- `meld_st_eng_jpn_jpn_all.csv` - All Japanese splits combined for ENG_JPN

## CSV Format

Each CSV file contains the following columns:
- `file`: Relative path to audio file (e.g., "MELD-ST/ENG_JPN/JPN/train/train_0.wav")
- `emotion`: Emotion label (anger, disgust, fear, joy, neutral, sadness, surprise)
- `sentiment`: Sentiment label (negative, neutral, positive)
- `speaker`: Speaker name/ID
- `language`: Language code (ENG, JPN, or DEU)
- `language_pair`: Language pair identifier (ENG_DEU or ENG_JPN)

**Note**: The audio files are referenced relative to the `audio_path` specified in the experiment configuration.

## Usage

### Preprocessing
Run the preprocessing script to generate CSV files:
```bash
cd data/meld-st
python preprocess_meld_st.py
```

Or for specific language pairs:
```bash
python preprocess_meld_st.py --language_pairs ENG_JPN
python preprocess_meld_st.py --language_pairs ENG_DEU
```

### Running Experiments
From the nkululeko root directory, run the emotion recognition experiments:

#### English Speech Emotion Recognition (ENG_JPN dataset)
```bash
python -m nkululeko.nkululeko --config data/meld-st/exp_eng_jpn_eng.ini
```

#### Japanese Speech Emotion Recognition (ENG_JPN dataset)
```bash
python -m nkululeko.nkululeko --config data/meld-st/exp_eng_jpn_jpn.ini
```

#### For German dataset (if available)
```bash
python -m nkululeko.nkululeko --config data/meld-st/exp_eng_deu_deu.ini
python -m nkululeko.nkululeko --config data/meld-st/exp_eng_deu_eng.ini
```

### Examples of Configuration Files
The repository includes pre-configured experiment files:
- `exp_eng_jpn_eng.ini` - English emotion recognition using ENG_JPN dataset
- `exp_eng_jpn_jpn.ini` - Japanese emotion recognition using ENG_JPN dataset

### Using with nkululeko
Example configuration for emotion recognition:
```ini
[DATA]
databases = ['train', 'dev', 'test']
audio_path = ./data/meld-st/
train = ./data/meld-st/meld_st_eng_jpn_eng_train.csv
train.type = csv
train.absolute_path = False
dev = ./data/meld-st/meld_st_eng_jpn_eng_valid.csv  
dev.type = csv
dev.absolute_path = False
test = ./data/meld-st/meld_st_eng_jpn_eng_test.csv
test.type = csv
test.absolute_path = False
target = emotion
```

## Original Dataset

The MELD-ST dataset should be downloaded and placed in the `MELD-ST/` subdirectory with the following structure:
```
MELD-ST/
├── ENG_DEU/
│   ├── deu_train.csv
│   ├── deu_valid.csv
│   ├── deu_test.csv
│   ├── ENG/
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   └── DEU/
│       ├── train/
│       ├── dev/
│       └── test/
└── ENG_JPN/
    ├── jpn_train.csv
    ├── jpn_valid.csv
    ├── jpn_test.csv
    ├── ENG/
    │   ├── train/
    │   ├── dev/
    │   └── test/
    └── JPN/
        ├── train/
        ├── dev/
        └── test/
```

## Text emotion recognition  
To run experiments on text emotion recognition, you must have "text" column (default). The header column is not necessary named "text", you can use Nkululeko mapping or specify like below.

```ini
[FEATS]
type = ['bert']
bert.model = line-corporation/line-distilbert-base-japanese
bert.text_column = Japanese2  # if text column is not "text"
```
Then run it with `nkululeko.nkululeko`. Example of results are below; highlights showed that it uses Japanese LINE distillbert instead of Google Bert.

```bash
...
DEBUG: featureset: loading line-corporation/line-distilbert-base-japanese model...
DEBUG: featureset: value for bert.layer is not found, using default: 0
DEBUG: featureset: using hidden layer #6
The repository line-corporation/line-distilbert-base-japanese contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/line-corporation/line-distilbert-base-japanese .
 You can inspect the repository content at https://hf.co/line-corporation/line-distilbert-base-japanese.
You can avoid this prompt in future by passing the argument `trust_remote_code=True`.

Do you wish to run the custom code? [y/N] y
initialized line-corporation/line-distilbert-base-japanese model on cuda
DEBUG: featureset: extracting line-corporation/line-distilbert-base-japanese embeddings, this might take a while...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1008/1008 [00:03<00:00, 281.54it/s]
DEBUG: experiment: All features: train shape : (9077, 768), test shape:(1008, 768)
DEBUG: experiment: scaler: False
DEBUG: runmanager: run 0 using model mlp
DEBUG: model: value for n_jobs is not found, using default: 8
DEBUG: model: value for random_state is not found, using default: 23
DEBUG: model: seeding random to 23
DEBUG: model: value for loss is not found, using default: cross
DEBUG: model: using model with cross entropy loss function
DEBUG: model: value for device is not found, using default: cuda
DEBUG: model: using layers {'l2':64, 'l1':32}
DEBUG: model: init: training with dropout: 0.3
DEBUG: model: value for learning_rate is not found, using default: 0.0001
DEBUG: modelrunner: run: 0 epoch: 0: result: test: 0.229 UAR
DEBUG: modelrunner: run: 0 epoch: 1: result: test: 0.266 UAR
...
```

There are two examples of INI file available under this meld-st data directory for Japanese and Deutsche Text Emotion Recognition with UAR 0.315 and 0.295 accordingly. To compute English UAR, just use `bert.text_column = English` in those INI file.

## Reference:  
[1] Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 527–536. https://doi.org/10.18653/v1/p19-1050  
[2] Chen, S., Yahata, S., Shimizu, S., Yang, Z., Li, Y., Chu, C., & Kurohashi, S. (2024). MELD-ST: An Emotion-aware Speech Translation Dataset. Findings of the Association for Computational Linguistics ACL 2024, 10118–10126. https://doi.org/10.18653/v1/2024.findings-acl.601
