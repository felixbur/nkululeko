# Nkululeko pre-processing for IndoWaveSentiment
 
 ## Dataset description  
IndoWaveSentiment is an audio dataset designed for classifying emotional expressions in Indonesian speech. The dataset was created using recordings from 10 actors, equally split between men and women. Recordings were made using a mono channel cardioid vocal microphone positioned no more than 10 cm from the speakers, connected to a laptop or computer. The audio was captured at a sample rate of 16 kHz with 32-bit depth. Each actor repeated the same sentence three times across five emotional categories: neutral, happy, surprised, disgusted, and disappointed. The dataset comprises 300 .wav files in total. This data was utilized to develop deep learning models as an initial study on audio classification in multimodal sentiment analysis. IndoWaveSentiment is also suitable for various signal processing applications, including voice emotion classification, and supports the development of sentiment analysis.  

## File Naming Convention

| Component | Code | Description |
|-----------|------|-------------|
| **Actors** | 01-10 | Odd numbers are male actors, even numbers are female actors |
| **Emotion Class** | 01 | Neutral |
| | 02 | Happy |
| | 03 | Surprise |
| | 04 | Disgust |
| | 05 | Disappointed |
| **Intensity** | 01 | Normal |
| | 02 | Strong |
| **Repetition** | 01 | First repetition |
| | 02 | Second repetition |
| | 03 | Third repetition |

### Example

A file named `07-03-02-03.wav` indicates:
- Actor: 07 (male)
- Emotion: 03 (surprise)
- Intensity: 02 (strong)
- Repetition: 03 (third)

Therefore, the actor is male (07), with surprise emotion (03), strong intensity (02), on the third repetition (03).

## Pre-processing command 
Download link: [https://data.mendeley.com/datasets/j9ytfdzy27/1](https://data.mendeley.com/datasets/j9ytfdzy27/1). The following commands assumed that data is download in current directory (`nkululeko/data/indowave`).  

```bash
# Extract the dataset
unzip 'IndoWaveSentiment Indonesian Audio Dataset for Emotion Classification.zip'

# Generate CSV files (train/dev/test splits)
python3 process_database.py

# Run experiment
cd ../..
python3 -m nkululeko.nkululeko --config data/indowave/exp.ini
```

The `process_database.py` script generates:
- `indowave_train.csv` - 180 samples (6 speakers: 3 male, 3 female)
- `indowave_dev.csv` - 60 samples (2 speakers: 1 male, 1 female)
- `indowave_test.csv` - 60 samples (2 speakers: 1 male, 1 female)
- `indowave.csv` - 300 total samples

Splits are speaker-independent to ensure generalization.

## Reference:  
[1] Bustamin, Anugrayani, Andi M. Rizky, Elly Warni, and Intan Sari Areni. "IndoWaveSentiment: Indonesian audio dataset for emotion classification." Data in Brief 57 (2024): 111138.  