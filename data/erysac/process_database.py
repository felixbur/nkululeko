# process_database.py for emochildru datasets

import argparse
from pathlib import Path

import pandas as pd

from nkululeko.utils.files import find_files

gender_map = {"m": "male", "f": "female"}

def process_database(data_dir, output_dir):
    # check if data_dir exists
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} not found")
    
    # check if output_dir exists, create if not
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # find all wav files 
    wavs = find_files(data_dir, ext=["wav"], relative=True)

    # extract filename as df
    df = pd.DataFrame({"file": wavs})

    # extract emotion from filename's parent
    df["emotion"] = df["file"].apply(lambda x: Path(x).parent.name.lower())

    # extract speaker first string before _ in the filename
    df["speaker"] = df["file"].apply(lambda x: Path(x).stem.split("_")[0])

    # extract gender, second string, m for male f for female, map
    df["gender"] = df["file"].apply(lambda x: Path(x).stem.split("_")[1])
    # map m to male, f to female
    df["gender"] = df["gender"].apply(lambda x: gender_map[x])

    # extract age, third string, remove the suffix y
    df["age"] = df["file"].apply(lambda x: Path(x).stem.split("_")[2][:-1])

    # check number of speakers
    speakers = df["speaker"].unique()
    print(f"Number of speakers: {len(speakers)}")

    # allocate 20% of speakers as test
    test_speakers = speakers[: len(speakers) // 5]
    df_test = df[df["speaker"].isin(test_speakers)]
    
    # allocate 20% train for dev
    dev_speakers = speakers[len(speakers) // 5 : len(speakers) // 5 * 2]
    df_dev = df[df["speaker"].isin(dev_speakers)]
    df_train = df.drop(df_dev.index)

    # save to CSV
    for split in ["train", "dev", "test"]:
        df_split = eval(f"df_{split}")
        df_split.to_csv(output_dir / f"erysac_{split}.csv", index=False)
        print(f"Saved {split} set to {output_dir / f'erysac_{split}.csv'}"
              f"with {len(df_split)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./Emotion-Recognition-of-Younger-School-Age-Children/")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    process_database(args.data_dir, args.output_dir)
