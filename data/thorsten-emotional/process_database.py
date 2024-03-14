# process_database.py --> Thorsten-Emotional

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path,
        default=Path('./thorsten-emotional_v02'))
    parser.add_argument('--output_dir', type=Path, default=Path('./'))
    args = parser.parse_args()

    # Read the database
    input_dir = args.data_dir
    output_dir = args.output_dir

    # list of all the WAV files in the database
    wav_list = list(input_dir.glob('**/*.wav'))
    
    file = [p for p in wav_list]
    emotion = [p.parts[1] for p in file]


    # convert to df
    df = pd.DataFrame(data={'file': file, 'emotion': emotion})

    # split to train, dev, test sets using scikit-learn
    train, test = train_test_split(df, test_size=0.2, 
        random_state=42, stratify=df['emotion'])

    # save to csv
    train.to_csv(output_dir / 'thorsten-emotional_train.csv', index=False)
    test.to_csv(output_dir / 'thorsten-emotional_test.csv', index=False)


if __name__ == '__main__':
    main()
