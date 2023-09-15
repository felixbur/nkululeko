# process_database.py --> MELD
# need to install mediainfo: apt-get install mediainfo
"""
This script writes the file paths, label, and other metadata of records to retain for model development in a csv.
The resultant csvs is used to make experiments with Nkululeko.
"""

import argparse
from collections import namedtuple
from os import walk
from os.path import exists
from pathlib import Path

import pandas as pd
from yaml import YAMLError, safe_load

LANG = "eng"  # ISO 639-3 English
LANG2 = "en-us"  # ISO 639-1 English + ISO 3166-1 United States
DATASET = "meld"

speaker_merge = {
    "Dr. Leedbetter": "Dr. Ledbetter",
    "A Waiter": "The Waiter",
    "Paleontologist": "Professore Clerk",
    "Ross and Joey": "Joey and Ross",
    "Both": "Joey and Chandler",
    "Phoebe Sr": "Phoebe Sr.",
    "Rachel and Phoebe": "Phoebe and Rachel",
}

bad_index = {"dia49_utt5", "dia66_utt9", "dia66_utt10", "dia38_utt4",
    "dia27_utt0",
    "dia27_utt1",
    "dia71_utt1",
    "dia93_utt5",
    "dia93_utt6",
    "dia93_utt7",
    "dia108_utt1",
    "dia108_utt2",
    "dia125_utt3", "dia4_utt1", "dia503_utt10", "dia715_utt0"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="MELD/MELD.Raw")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    with open(f"{data_dir}/speaker_genders.txt", "r") as f:
        speaker_genders = {line.split("\t")[0]: line.strip()[-1] for line in f}

    splits = ["train", "dev", "test"]
    split_dirs = ["train/train_splits", "dev_splits_complete",
                  "test/output_repeated_splits_test"]
    csvs = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]

    # name_map = {}
    # all_paths = []
    # for split, x in zip(splits, split_dirs):
    #     paths = list(data_dir.glob(f"{x}/dia*.mp4"))
    #     all_paths.extend(paths)
        # name_map.update(
        #     {x: data_dir / f"{split}_{x.stem}.wav" for x in paths})
    dfs = []
    for split, x in zip(splits, csvs):
        df = pd.read_csv(data_dir / x)
        df.index = df.apply(
            lambda x: f"dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}", axis=1
        )
        if split == "train":
            # Train dia125_utt3 is broken
            df = df.drop("dia125_utt3")
        df["split"] = split
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.drop(index={"dia110_utt7"})
    df = df.drop(index=bad_index)

    # merger similar speakers
    df["Speaker"] = df["Speaker"].replace(speaker_merge)

    # get gender from speaker
    df["gender"] = df["Speaker"].map(speaker_genders).replace({
        "m" : "male", "f": "female"
    })

    # rename columns
    df = df.rename(columns={"Emotion": "emotion", "Sentiment": "sentiment", "Speaker": "speaker"})

    # set index to file name
    df.index.name = "file"

    # add .mp4 extension to file name
    df.index = df.index + ".wav"

    # save train, dev, test splits to separate csv
    df.to_csv(output_dir / f"{DATASET}.csv")
    df[df["split"] == "train"].to_csv(output_dir / f"{DATASET}_train.csv")  
    df[df["split"] == "dev"].to_csv(output_dir / f"{DATASET}_dev.csv")
    df[df["split"] == "test"].to_csv(output_dir / f"{DATASET}_test.csv")

    print(f"Total: {len(df)}, Train: {len(df[df['split'] == 'train'])}, Dev: {len(df[df['split'] == 'dev'])}, Test: {len(df[df['split'] == 'test'])}")

    # print(f"emotion: {df['emotion'].unique()}, number of emotions: {len(df['emotion'].unique())}")
    # print column and index attributes
    # print(df.columns)   

if __name__ == "__main__":
    main()
