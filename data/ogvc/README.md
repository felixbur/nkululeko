# Nkululeko pre-processing for OGVC dataset

OGVC (Online Gaming Voice Chat) is an acted and natural emotional speech database containing recordings from multiple speakers expressing various emotions with different intensity levels [1].

## Dataset Structure

The dataset contains emotional speech recordings organized into the following structure:
- **Speakers**: 4 speakers (F1, F2, M1, M2) where F = Female, M = Male
- **Emotions**: 8 emotional categories
  - JOY (Joy/Happiness)
  - ANG (Anger) 
  - SAD (Sadness)
  - ANT (Anticipation)
  - FEA (Fear)
  - SUR (Surprise)
  - ACC (Acceptance/Neutral)
  - DIS (Disgust)
- **Intensity levels**: 4 levels (0, 1, 2, 3) representing different emotional intensities
- **Gender**: Both male and female speakers

## Data Splits

The dataset is split based on speaker assignment:
- **Training set**: F1 and M1 speakers
- **Development set**: F2 speaker  
- **Test set**: M2 speaker

## File Naming Convention

Audio files follow a specific naming pattern:
- Format: `[GENDER][SPEAKER_ID][TEXT_ID][EMOTION][INTENSITY].wav`
- Example: `FOY0101ANT0.wav` = Female speaker, text 0101, Anticipation emotion, intensity level 0

## Processing

The database can be processed using the provided script:

```bash
# Process the raw audio files to generate CSV metadata files
python process_database.py --data_dir /path/to/wav/files --output_dir ./

# This will generate:
# - ogvc_train.csv (training split)
# - ogvc_dev.csv (development split) 
# - ogvc_test.csv (test split)
```

## Usage with Nkululeko

After processing, you can run experiments using the provided configuration:

```bash
# Run emotion recognition experiment
python -m nkululeko.nkululeko data/ogvc/exp.ini
```

The experiment configuration (`exp.ini`) is set up for:
- Multi-class emotion classification (8 emotions)
- HuBERT-XLarge features
- SVM classifier
- Standard feature scaling

## Dataset Statistics

The processed CSV files contain the following columns:
- `file`: Path to the audio file
- `gender`: Speaker gender (male/female)
- `emotion`: Emotion label (JOY, ANG, SAD, ANT, FEA, SUR, ACC, DIS)
- `intensity`: Intensity level (0-3)

## Notes

- All audio files are in WAV format
- The dataset appears to be structured for acted emotional speech research
- Intensity levels provide additional granularity for emotion recognition tasks
- Speaker-independent splits ensure proper evaluation of model generalization

## References

[1] https://research.nii.ac.jp/src/en/OGVC.html