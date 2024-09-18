#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# process_database.py --> OREAU2
import argparse
from pathlib import Path

import pandas as pd

from nkululeko.utils import files

emotion_map = {
    "C": "anger",
    "D": "disgust",
    "J": "happiness",
    "N": "neutral",
    "P": "fear",
    "S": "surprise",
    "T": "sadness",
}

gender_map = {
    "f": "female",
    "m": "male"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="OREAU2")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob("**/*.wav"))
    files = [file for file in paths]
    emotions = [emotion_map[file.stem[-2]] for file in files]
    speakers = [file.stem[:2] for file in files]
    genders = [gender_map[file.parts[-3]] for file in files]
    languages = ["french" for file in files]

    # convert to df
    df = pd.DataFrame({"file": files, "emotion": emotions, "speaker": speakers, "gender": genders, "language": languages})

    # allocate speakers <= 11 for test
    df_train = df[df["speaker"] > "11"]
    df_test = df.drop(df_train.index)

    # save to csv
    df.to_csv(output_dir / "oreau2.csv", index=False)
    df_train.to_csv(output_dir / "oreau2_train.csv", index=False)
    df_test.to_csv(output_dir / "oreau2_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")


if __name__ == "__main__":
    main()
