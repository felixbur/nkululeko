# process_database.py --> ESD

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

emotion_map = {
    'Neutral' : "neutral", 
    'Sad': "sadness",
    'Surprise': "surprise", 
    'Angry': "anger",
    'Happy': "happiness"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="ESD")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load data
    wav_files = list(data_dir.glob("**/*.wav"))
    
    file = [f for f in wav_files]
    emotion = [f.parent.name for f in wav_files]
    speaker = [f.parts[1] for f in wav_files]
    language = ['chinese' if int(f.parts[1]) <=10 else 'english' for f in wav_files]
    gender = ['female' if int(f.parts[1] in [ [1, 2, 3, 7, 9, 12, 15, 16, 17, 18]])
              else 'male' for f in wav_files]

    # Create dataframe
    df = pd.DataFrame({'file': file, 'emotion': emotion, 'speaker': speaker, 
                       'language': language, 'gender': gender})

    # Split into training and test sets, set speaker 9,10 and 18,20 aside for test
    test_df = df[df['speaker'].isin(['0009', '0010', '0018', '0020'])]
    # drop the rest for test
    train_df = df.drop(test_df.index)

    # save to csv
    train_df.to_csv(output_dir / "esd_train.csv", index=False)
    test_df.to_csv(output_dir / "esd_test.csv", index=False)

if __name__ == "__main__":
    main()