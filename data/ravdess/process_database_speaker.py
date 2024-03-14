"""
This script is to import the 
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
database to nkululeko.

I used the version downloadable from [Zenodo](https://zenodo.org/record/1188976)

Download and unzip the file Audio_Speech_Actors_01-24.zip e.g., `ravdess_speech`

adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition

Usage: `python3 process_database.py -d /data/ravdess_speech
OR
`python3 process_database.py /data/ravdess_speech`
"""

 
import argparse
import os
import sys
from pathlib import Path, PurePath

import pandas as pd

# ravdess source directory as argument
# source_dir = './'
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, default='./', help='path to RAVDESS speech directory')
args = parser.parse_args()
source_dir = str(Path(args.dir))
database_name = 'ravdess'

# check if directory (e.g., Actor_01) exists
if not os.path.exists(os.path.join(source_dir, "Actor_01")):
    print(f'Error: {source_dir}/Actor_01 does not exist')
    raise FileNotFoundError

ravdess_directory_list = os.listdir(source_dir)
# print(ravdess_directory_list)

file_emotion = []
file_speaker = []
file_gender = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    if not dir.startswith('Actor'):
        continue
    if os.path.isdir(os.path.join(source_dir, dir)):
        actor = os.listdir(os.path.join(source_dir, dir))
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_gender.append(int(part[6]))
            file_speaker.append(str(part[6]))
            file_path.append(source_dir + '/' + dir + '/' + file)
        
# put all in one dataframe
result_df = pd.DataFrame(list(zip(file_path, file_emotion,file_speaker,file_gender)), columns=['file','emotion', 'speaker', 'gender'])

# change speaker labels to string
# result_df['speaker'] = result_df['speaker'].astype('str')

# add `spk` to speaker id
result_df['speaker'] = 'spk' + result_df['speaker'].astype('str')

# changing integers to actual emotions.
result_df.emotion.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

# changing numbers to gender designations
def get_gender(x):
    if x%2==0:
        return 'female'
    else:
        return 'male'
result_df.gender = result_df.gender.map(lambda x: get_gender(x))

train_df, test_df = result_df.groupby('speaker').apply(lambda x: x.sample(frac=0.8)).reset_index(drop=True), result_df.groupby('speaker').apply(lambda x: x.drop(x.sample(frac=0.8).index)).reset_index(drop=True)

# save to csv
train_df.to_csv(f'{database_name}_speaker_train.csv', index=False)
test_df.to_csv(f'{database_name}_speaker_test.csv', index=False)

# print length of train and test set and total
print(f"Total length: {len(result_df)}, Train set: {len(train_df)}, Test set: {len(test_df)}")

