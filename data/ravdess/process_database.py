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

 
import os
import pandas as pd
import sys
import argparse
from pathlib import Path, PurePath

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
print(ravdess_directory_list)

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


# changing integers to actual emotions.
result_df.emotion.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
# changing numbers to gender designations
def get_gender(x):
    if x%2==0:
        return 'female'
    else:
        return 'male'
result_df.gender = result_df.gender.map(lambda x: get_gender(x))

# For a gender balanced split close to 60%-20%-20%: train - 16, dev - 4 , test - 4
speaker_splits = {
    'train': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '18'],
    'dev': ['16', '17', '19', '20'],
    'test': ['21', '22', '23', '24']
}

# set the index to file paths
result_df = result_df.set_index('file')

# split the data
for split in ['train','test','dev']:
    split_df = result_df[result_df.speaker.isin(speaker_splits[split])]
    split_df.to_csv(f'{database_name}_{split}.csv')
    print(f'{split} split #samples: {split_df.shape[0]}')


# make a convinience file with all data
result_df.to_csv(database_name+'.csv')


"""
Should result into something like
file,emotion,speaker,gender
./Actor_09/03-01-08-01-02-02-09.wav,surprise,ravdess_09,male
./Actor_09/03-01-08-02-01-01-09.wav,surprise,ravdess_09,male
...
"""
