#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @filename: process_database.py
# @description: Process the ASED database 


import argparse
import shutil
from importlib.resources import path
from pathlib import Path

import pandas as pd
from joblib import delayed

emotion_map = {
    "a": "anger",
    "h": "happiness",
    "n": "neutral",
    "f": "fear",
    "s": "sadness",
}


def main():
    parser = argparse.ArgumentParser(
        usage="python3 process_database.py database output"
    )
    parser.add_argument("--data_dir", type=str, default="ASED_V1", help="Path to the ASED database")
    parser.add_argument("--out_dir", type=str, default=".", help="Path to the output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    paths = list(data_dir.glob("**/*.wav"))
    emotions = [emotion_map[path.stem[0].lower()] for path in paths]
    genders = [["female", "male"][int(p.stem[9:11]) - 1] for p in paths]
    spekaers = [p.stem[-2:] for p in paths]
    languages = ["amharic" for p in paths]

    # convert to df
    df = pd.DataFrame({"file": paths, "emotion": emotions, "gender": genders, "speaker": spekaers, "language": languages})

    # allocate speakers >= 55 for test
    df_test = df[df["speaker"] > "55"]
    df_train = df.drop(df_test.index)


    # save to csv
    df_train.to_csv(out_dir / "ased_train.csv", index=False)
    df_test.to_csv(out_dir / "ased_test.csv", index=False)
    df.to_csv(out_dir / "ased.csv", index=False)


if __name__ == "__main__":
    main()
    