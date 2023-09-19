#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : process_database.py --> MESS

import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "A": "anger",
    "C": "calm",
    "H": "happiness",
    "S": "sadness",
}

gender_map = {
    "M": "male",
    "F": "female"
}

def main():
    parser = argparse.ArgumentParser(usage="python3 process_database.py database output")
    parser.add_argument("--data_dir", type=str, default="MESS", help="Path to the RAVDESS database")
    parser.add_argument("--out_dir", type=str, default=".", help="Path to the output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    paths = list(data_dir.glob("**/*.wav"))
    emotions = [emotion_map[path.stem[0]] for path in paths]
    genders = [gender_map[p.stem[1]] for p in paths]
    speakers = [p.stem[1:3] for p in paths]
    languages = ["english" for p in paths]

    # convert to df
    df = pd.DataFrame({"file": paths, "emotion": emotions, "gender": genders, "speaker": speakers, "language": languages})

    # set speaker M3 as test
    df_test = df[df["speaker"] == "M3"]
    df_train = df.drop(df_test.index)

    # save to csv
    df_train.to_csv(out_dir / "mess_train.csv", index=False)
    df_test.to_csv(out_dir / "mess_test.csv", index=False)
    df.to_csv(out_dir / "mess.csv", index=False)

    print(f"Processed {len(df)} files, {len(df_train)} for training, {len(df_test)} for testing")


if __name__ == "__main__":
    main()
