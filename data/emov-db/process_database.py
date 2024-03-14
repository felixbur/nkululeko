#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# process_database.py --> EMOV-DB

import argparse
import re
from pathlib import Path

import pandas as pd
from nkululeko.utils import files

emotion_map = {
    "amused": "amusement",
    "anger": "anger",
    "disgust": "disgust",
    "neutral": "neutral",
    "sleepiness": "sleepiness",
}

gender_map = {"bea": "female", "jenie": "female", "josh": "male", "sam": "male"}


def main():
    parser = argparse.ArgumentParser(description="Process database")
    parser.add_argument("--data_dir", type=str, default="EmoV-DB_sorted")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # paths_list = files.find_files(data_dir, ext="wav")
    paths = list(data_dir.glob("**/*.wav"))
    # paths = [Path(p) for p in paths_list]

    emotions = [emotion_map[p.stem.split("_")[0].lower()] for p in paths]
    speakers = [p.parts[-3] for p in paths]
    genders = [gender_map[s] for s in speakers]
    languages = ["english" for file in paths]

    # convert to df
    df = pd.DataFrame(
        {
            "file": paths,
            "emotion": emotions,
            "speaker": speakers,
            "gender": genders,
            "language": languages,
        }
    )

    # allocate speaker sam for test
    train_df = df[df["speaker"] != "bea"]
    test_df = df.drop(train_df.index)

    # save to csv
    df.to_csv(output_dir / "emov-db.csv", index=False)
    train_df.to_csv(output_dir / "emov-db_train.csv", index=False)
    test_df.to_csv(output_dir / "emov-db_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
