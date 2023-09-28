#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# process_database.py --> JL

import argparse
import re
from pathlib import Path

import pandas as pd

REGEX = re.compile(r"^(?:fe)?male[12]_([a-z]+)_\d+[ab]_[12]$")


emotion_map = {
    "angry": "anger",
    "sad": "sadness",
    "neutral": "neutral",
    "happy": "happiness",
    "excited": "happiness",   # merge excited and happy
    "anxious": "anxiety",
    "apologetic": "apologetic",
    "assertive": "assertive",
    "concerned": "concern",
    "encouraging": "encouraging",
}

secondary_emotions = ["anxious", "apologetic",
                      "assertive", "concerned", "encouraging"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="JL")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(
        data_dir.glob("Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav")
    )
    # print p.stem with NoneType
    pstem = [p.stem for p in paths if REGEX.match(p.stem) is None]
    emotions = [emotion_map.get(REGEX.match(p.stem).group(1)) for p in paths]
    speakers = [p.stem[: p.stem.find("_")] for p in paths]
    genders = [speaker[:-1] for speaker in speakers]
    languages = ["english" for p in paths]
    countries = ["new zealand" for p in paths]

    # convert to df
    df = pd.DataFrame({"file": paths, "emotion": emotions, "speaker": speakers, "gender": genders, "language": languages, "country": countries})

    # allocate speaker male2 for test
    df_train = df[df["speaker"] != "male2"]
    df_test = df.drop(df_train.index)

    # save to csv
    df.to_csv(output_dir / "jl.csv", index=False)
    df_train.to_csv(output_dir / "jl_train.csv", index=False)
    df_test.to_csv(output_dir / "jl_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")


if __name__ == "__main__":
    main()
    
