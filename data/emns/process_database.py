#!/usr/bin/env python3
# process_database.py --> EMNS

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

emotion_map = {
    'Sarcastic': 'sarcastic',
    'Excited': 'happiness',         # merge excited with happiness
    'Neutral': 'neutral',
    'Surprised': 'surprise',
    'Disgust': 'disgust',
    'Sad': 'sadness',
    'Angry': 'anger',
    'Happy': 'happiness'
}

def main():
    parser = argparse.ArgumentParser(description='Process database')    
    parser.add_argument('--data_dir', type=str, default='EMNS', help='data directory')
    parser.add_argument('--metadata_file', type=str, default='EMNS/metadata.csv', help='metadata file')
    parser.add_argument('--output_dir', type=str, default='.', help='data file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata_file = Path(args.metadata_file)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()

    df = pd.read_csv(metadata_file, delimiter='|')

    # male all lowercase
    df = df.rename(columns={'audio_recording': 'file', 'user_id': 'speaker'})

    # remove wavs path from file names
    df['file'] = df['file'].str.replace('wavs/', '')
    
    # make gender lowercase
    df['gender'] = df['gender'].str.lower()

    # map emotions
    df['emotion'] = df['emotion'].map(emotion_map)

    # split into train and test based on emotion
    train, test = train_test_split(df, test_size=0.2, stratify=df['emotion'])

    # save to csv
    train.to_csv(output_dir / 'emns_train_webm.csv', index=False)
    test.to_csv(output_dir / 'emns_test_webm.csv', index=False)

    # save to csv with WAV extension
    train['file'] = train['file'].str.replace('.webm', '.wav')
    train.to_csv(output_dir / 'emns_train.csv', index=False)
    test['file'] = test['file'].str.replace('.webm', '.wav')
    test.to_csv(output_dir / 'emns_test.csv', index=False)

    print(f"Total: {len(df)}, Train: {len(train)}, Test: {len(test)}")


if __name__ == '__main__':
    main()
