import argparse
import os
import sys
from pathlib import Path, PurePath

import pandas as pd

# ravdess source directory as argument
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, default='./02_process', help='split train, dev, test dataset')
args = parser.parse_args()
source_dir = str(Path(args.dir))
database_name = 'en-dialect'


my_csv_path = 'en-dialect_all_add_path.csv'

# check for the csv file
if not os.path.exists(my_csv_path):
    print(f'error: your csv path {my_csv_path} not exist')
    raise FileNotFoundError

# load csv / 
# print(f'loading from {my_csv_path} ...')
result_df = pd.read_csv(my_csv_path)


#split into train dev test
# set the portion
train_frac = 0.8
dev_frac = 0.1
test_frac = 0.1

# Temporary variable
rest_df = pd.DataFrame() 
train_df = pd.DataFrame()

# Step 1: Group speakers and allocate 80% of the data to the training set.
print("dividing training set...")
for speaker in result_df['speaker'].unique():
    speaker_df = result_df[result_df['speaker'] == speaker]
    train_sample = speaker_df.sample(frac=train_frac, random_state=42) 
    train_df = pd.concat([train_df, train_sample])
    
    rest_sample = speaker_df.drop(train_sample.index)
    rest_df = pd.concat([rest_df, rest_sample])

# Step 2: Divide the remaining data equally into a develop set and a test set.
print("divding develop and test set")
dev_split_frac = dev_frac / (dev_frac + test_frac)

dev_df = pd.DataFrame()
test_df = pd.DataFrame()

for speaker in rest_df['speaker'].unique():
    speaker_df = rest_df[rest_df['speaker'] == speaker]
    dev_sample = speaker_df.sample(frac=dev_split_frac, random_state=42)
    dev_df = pd.concat([dev_df, dev_sample])

    test_sample = speaker_df.drop(dev_sample.index)
    test_df = pd.concat([test_df, test_sample])



# save files
train_df.to_csv(f'{database_name}_train.csv', index=False)
dev_df.to_csv(f'{database_name}_dev.csv', index=False)
test_df.to_csv(f'{database_name}_test.csv', index=False)

# print length to check
total_len = len(result_df)
train_len = len(train_df)
dev_len = len(dev_df)
test_len = len(test_df)
print("\nSplit complete!")
print(f"Total sample size: {total_len}")
print(f"training set: {train_len} ({train_len/total_len:.0%})")
print(f"develop set: {dev_len} ({dev_len/total_len:.0%})")
print(f"text set: {test_len} ({test_len/total_len:.0%})")