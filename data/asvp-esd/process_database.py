# process_database --> ASVP-ESD

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

emotion_map = {
    '01': 'boredom',
    '02': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'anger',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise',
    '09': 'excited',
    '10': 'pleasure',
    '11': 'pain',
    '12': 'disapointed',
    '13': 'others',
}


vocal_map = {
    "01" : "speech",
    "02" : "non_speech",
}

language_map = {
    "00" : "chinese",
    "01" : "english",
    "02" : "french",
    "03" : "russian",
    "04" : "other1",
    "05" : "other2",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path,
        default=Path('./ASVP-ESD-Update'))
    parser.add_argument('--output_dir', type=Path, default=Path('./'))
    args = parser.parse_args()

    # Read the database
    input_dir = args.data_dir
    output_dir = args.output_dir

    # list of all the WAV files in the database
    wav_list = list(input_dir.glob('Audio/**/*.wav'))

    # file = [p for p in wav_list if os.path.getsize(p) >= 40000]
    file = [p for p in wav_list]

    # Vocal channel (01 = speech, 02 = non speech).
    vocal = [vocal_map[p.stem.split('-')[1]] for p in file]

    # emotion
    emotion = [emotion_map[p.stem.split('-')[2]] for p in file]

    # language --> not all data have language
    # language = [language_map[p.stem.split('-')[8]] for p in wav_list]

    # save to pandas dataframe
    df = pd.DataFrame(data={'file': file, 'vocal': vocal, 'emotion': emotion})

    # split to train, test sets using scikit-learn

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # save to csv
    train.to_csv('asvp_train.csv', index=False)
    test.to_csv('asvp_test.csv', index=False)

if __name__ == '__main__':
    main()
