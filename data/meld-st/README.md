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
- Train: 9,314 samples./data/meld-st/MELD-ST/ENG_DEU/DEU
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

#### Japanese Sepeech Emotion Recognition (ENG_JPN dataset)
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

## Exlcluded files  
After running inital experiment on ENG_DEU, we found some files have zero size; hence we removed from the list using Nkululeko `check_size=5000` (under [DATA] section).

## Reference:  
[1] Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 527–536. https://doi.org/10.18653/v1/p19-1050