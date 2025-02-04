# process_database.py: pre-processing script for JNV database

import argparse
import os

import numpy as np
import pandas as pd


def read_audio_files(data_dir):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = file.split("_")[1]
                data.append({"file": os.path.join(
                    root, file), "emotion": emotion})

    df = pd.DataFrame(data)
    return df


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir

    df = read_audio_files(data_dir)
    train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=42), [
                                         int(.8 * len(df)), int(.9 * len(df))])

    # print number of files in each set
    print(f"Number of files in train set: {len(train_df)}")
    print(f"Number of files in dev set: {len(dev_df)}")
    print(f"Number of files in test set: {len(test_df)}")
    print(f"Number of files in total: {len(df)}")

    train_df.to_csv(os.path.join(output_dir, "jnv_train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "jnv_dev.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "jnv_test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="JNV/",
        help="Directory containing audio files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Output dir for CSV files")
    args = parser.parse_args()

    main(args)
