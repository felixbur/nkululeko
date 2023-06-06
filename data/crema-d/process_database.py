"""
This folder is to import the 
The Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)
database to nkululeko.

I used the version downloadable from [github](https://github.com/CheyneyComputerScience/CREMA-D)

downloaded April 27th 2023

Download and unzip or git clone the archive

I only used the AudioWAV subdirectory

adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition

Usage: `python3 process_database_split.py /data/Crema-D/AudioWAV`

"""

import pandas as pd
import os
import sys

# Crema-D source dir as argument
# Crema = './AudioWAV'
Crema = sys.argv[1]
dataset_name = 'crema-d'


crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
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
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df = Crema_df.set_index('file')
Crema_df.head()
Crema_df.to_csv(f'{dataset_name}.csv')