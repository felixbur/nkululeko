# process_database.py: pre-processing script for JNV database

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_audio_files(data_dir):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = file.split("_")[1]
                data.append({"file": os.path.join(root, file), "emotion": emotion})
    
    df = pd.DataFrame(data)
    return df

def main(args):
    data_dir = args.data_dir
    output_file = args.output_file
    
    df = read_audio_files(data_dir)
    train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
    
    train_df.to_csv("jnv_train.csv", index=False)
    dev_df.to_csv("jnv_dev.csv", index=False)  
    test_df.to_csv("jnv_test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="jnv_corpus_ver2/JNV/", help="Directory containing audio files")
    parser.add_argument("--output_file", type=str, default="jnv_database.csv", help="Output CSV file")
    args = parser.parse_args()
    
    main(args)
