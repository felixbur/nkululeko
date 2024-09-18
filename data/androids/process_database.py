"""
This folder is to import the 
Androids-corpus depression
database to nkululeko.

I used the version downloadable from [github](https://github.com/CheyneyComputerScience/CREMA-D)

downloaded April 27th 2023

I used the version downloadable from [Dropbox, mentioned in this github page](https://github.com/androidscorpus/data)

Download and unzip the file Androids-corpus.zip to the current folder

Usage: `python process_database.py`

"""

import os

import audeer
import pandas as pd

dataset_name = 'androids'
data_root = './Androids-Corpus/'

# read in the fold list 
fold_dict = {}
df_fold = pd.read_csv(f'{data_root}fold-lists.csv')
for i in range(df_fold.shape[0]): 
    for j in range(df_fold.shape[1]): 
        try:
            value = df_fold.iloc[i, j].replace('\'', '')
        except AttributeError:
            value = 'na' 
        if j in range(7, 12):
            fold_dict[value] = j - 7
        else:
            fold_dict[value] = j 
directory_list = audeer.list_file_names(data_root, filetype='wav', recursive=True, basenames=True)

depressions, speakers, educations, genders, ages, tasks, folds = [], [], [], [], [], [], []
file_paths = []
print(len(directory_list))
gender_map = {'F':'female', 'M':'male'}
depression_map = {'P':'depressed', 'C':'control'}

for file in directory_list:
    # storing file paths
    # file = file.replace(os.getcwd(), '.')
    file_paths.append(data_root+file)
    # storing file emotions
    fn = audeer.basename_wo_ext(file)

    # The naming convention of the audio files is as follows:
    # nn_XGmm_t.wav
    # where nn is a unique integer identifier such that, in a given group, files with the same nn contain the voice of the same speaker (there is a trailing 0 for numbers lower than 10), X is an alphabetic character corresponding to the speaker’s condition (P for depression patient and C for control), G is an alphabetic character that stands for the speaker’s gender (M for male and F for female), mm is a two-digits integer number corresponding to the speaker’s age, and t is an integer number between 1 and 4 accounting for the education level (1 corresponds to primary school and 4 corresponds to university). The letter X was used for the 2 participants who did not provide information about this aspect. There is no indication of the task because recordings corresponding to RT and IT are stored in different directories.
    if 'Reading-Task'in file:
        task = 'reading'
    elif 'Interview-Task' in file:
        task = 'interview'
    else:
        print('ERROR: task undefined')
        exit(-1)
 
    part = fn.split('_')
    dir_name = f'{part[0]}_{part[1]}_{part[2]}'
    depression = part[1][0]
    speaker = f'{depression}_{part[0]}'
    gender = part[1][1]
    age = part[1][2:4]
    education = part[2]
    depressions.append(depression_map[depression])
    speakers.append(speaker)
    genders.append(gender_map[gender])
    ages.append(age)
    tasks.append(task)
    educations.append(education)
    folds.append(fold_dict[dir_name])
#    print(f'{file} {speaker}')


# dataframe for emotion of files
df = pd.DataFrame({'file':file_paths, 
                   'speaker':speakers, 
                   'gender':genders, 
                   'age':ages, 
                   'task':tasks, 
                   'depression':depressions, 
                   'education':educations, 
                   'fold':folds})

df = df.set_index('file')
df.head()
df.to_csv(f'{dataset_name}.csv')