"""
This folder is to import the 
The Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)
database to nkululeko.

I used the version downloadable from [github](https://github.com/CheyneyComputerScience/CREMA-D)

downloaded April 27th 2023

Download and unzip or git clone the archive

I only used the AudioWAV subdirectory

adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition

Usage: `python3 process_database.py` 
OR
`python3 --dir /data/Crema-D/AudioWAV`
help: `python3 process_database.py --help`

Output: crema-d.csv
columns: emotion, file, speaker_id, split
split is 0 for train, 1 for dev, 2 for test
"""

import pandas as pd
import os
import argparse
from pathlib import Path, PurePath
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

# Crema-D source dir as argument, default is './AudioWAV'
# Crema = './AudioWAV'
parser.add_argument('-d', '--dir', type=str, default='./AudioWAV', help='path to Crema-D AudioWAV directory')
args = parser.parse_args()
Crema = Path(args.dir)
dataset_name = 'crema-d'

# check if directory exists
if not Crema.exists():
    print(f'Error: {Crema} does not exist')
    exit()

# crema_directory_list = os.listdir(Crema)
crema_directory_list = [f for f in Crema.glob('*.wav') if f.is_file()]

file_emotion = []
file_path = []
file_speaker = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(file)
    # storing file speakers
    file_speaker.append(str(file.name).split('_')[0])
    # storing file emotions
    part=str(file).split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])
# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['file'])
# dataframe for speaker IDs
speaker_df = pd.DataFrame(file_speaker, columns=['speaker_id'])

Crema_df = pd.concat([emotion_df, path_df, speaker_df], axis=1)
Crema_df = Crema_df.set_index('file')

# Splitting data based on speaker ID
remaining_speakers, test_speakers = train_test_split(Crema_df['speaker_id'].unique(), test_size=0.2, random_state=42)
train_speakers, dev_speakers, = train_test_split(remaining_speakers, test_size=0.2, random_state=42)

Crema_df['split'] = 0 # Initialize split column with 'train' for all rows

# Assigning split values based on speaker ID
Crema_df.loc[Crema_df['speaker_id'].isin(dev_speakers), 'split'] = 1 #'validation'
Crema_df.loc[Crema_df['speaker_id'].isin(test_speakers), 'split'] =2 #'test'

# save csv for each partition
if Crema_df['split'].unique().size == 3:
    Crema_df.loc[Crema_df['split'] == 0].to_csv(f'{dataset_name}_train.csv')
    Crema_df.loc[Crema_df['split'] == 1].to_csv(f'{dataset_name}_dev.csv')
    Crema_df.loc[Crema_df['split'] == 2].to_csv(f'{dataset_name}_test.csv')
# Crema_df.to_csv(f'{dataset_name}.csv')

