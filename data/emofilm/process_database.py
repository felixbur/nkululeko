# process_database.py -> EmoFilm database

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# load data

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="./EmoFilm", 
                    help='path to data')
args = parser.parse_args()

data_excel = os.path.join(args.data, 'f_m_corpus_it_es_en_legend.xlsx')

label_female_df = pd.read_excel(data_excel,sheet_name="female")
label_male_df = pd.read_excel(data_excel,sheet_name="male")

# merge female and male df
label_df = pd.merge(label_female_df, label_male_df, how='outer')

# Use 'speaker' as the grouping object for cross-validation
groups = label_df["speaker"]

# get language from the file name
label_df['gender'] = label_df['file'].str.split('_').str[0]
label_df['language'] = label_df['file'].str[-2:]
label_df['emotion'] = label_df['file'].str[2:5]

# replace emotion labels with dictionary
emo_dict = {'ans': 'fea', 'dis': 'con', 'gio': 'hap', 'rab': 'ang', 'tri': 'sad'}
label_df['emotion'] = label_df['emotion'].map(emo_dict)

# change file to include full path
label_df['file'] = label_df['file'].apply(lambda x: f"{args.data}/wav_corpus_16k/{x}_16k.wav")

# split data into train, dev, test
# train: 80%, dev: 10%, test: 10%
train_files, test_files = train_test_split(label_df, test_size=0.2, random_state=42, stratify=label_df['emotion'])

# allocate 10% of train to dev
train_files, dev_files = train_test_split(train_files, test_size=0.125, random_state=42, stratify=train_files['emotion'])

# write to CSV
train_files.to_csv(f"emofilm_train.csv", index=False)
dev_files.to_csv(f"emofilm_dev.csv", index=False)
test_files.to_csv(f"emofilm_test.csv", index=False)

