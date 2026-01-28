# Nkululeko Pre-processing for E-SERAVD Dataset
 
## Dataset description  
The **E-SERAVD** is an Indonesian Speech Emotion Recognition (SER) corpus developed by the **Telecommunication Signal Processing Laboratory, EEPIS**.  
It consists of emotional speech segments extracted from Indonesian film dialogues, designed to support research in **speech emotion recognition**, **multimodal emotion analysis**, and **affective computing**.

Each clip labeled with **"T" (Tested)** has undergone a full annotation and statistical validation process to ensure **high reliability** and **strong inter-rater agreement**.  
The dataset includes both **Audio-Only (.wav)** and **Audio-Video (.mp4)** versions to facilitate comprehensive emotion recognition research and model evaluation.  

### Dataset Statistics
- **Total samples**: 1,200 wav files
- **Speakers**: 6 (AADC, D1990, DHB, HA3, HSL, TAOL)
- **Emotions**: 4 classes - angry (300), sad (300), neutral (300), happy (300)
- **Language**: Indonesian
- **Source**: Film dialogue extracts

### File Naming Convention
Files follow the format: `02-<emotion>-<speaker>-T-<number>.wav`
- `02`: Dataset version/prefix
- `<emotion>`: 01=angry, 02=sad, 03=neutral, 04=happy
- `<speaker>`: AADC, D1990, DHB, HA3, HSL, TAOL
- `T`: Tested/validated annotation
- `<number>`: Sample ID (0001-XXXX)

## Processing command  

### Download and extract dataset
Download the dataset from: wget https://e.pcloud.link/publink/show?code=XZd62aZSxD3MBAoi8VRhOgKj0n7LLvwzN0y, then follows steps to extract:

```bash
unzip E-SERAVD.zip
```

### Generate CSV files
```bash
cd data/e-seravd
python3 process_database.py
```

This will generate:
- `e-seravd_train.csv` - 679 samples (3 speakers: DHB, HA3, HSL)
- `e-seravd_dev.csv` - 181 samples (1 speaker: TAOL)
- `e-seravd_test.csv` - 340 samples (2 speakers: AADC, D1990)
- `e-seravd.csv` - 1,200 total samples

### Run experiment
```bash
cd ../..
python3 -m nkululeko.nkululeko --config data/e-seravd/exp.ini
```

## Results
Using OpenSmile features + XGBoost:
- **UAR**: 0.287 (28.7%)
- **Accuracy**: 29.1%
- **Note**: Low performance suggests challenging dataset with significant speaker variability and overlapping emotion characteristics in film dialogues
