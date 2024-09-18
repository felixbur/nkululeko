#!/usr/bin/env python3
# process_database.py --> MLEndSND

import argparse
from pathlib import Path

import pandas as pd

from nkululeko.utils import files


def main():
    parser = argparse.ArgumentParser(description="Process database")
    parser.add_argument("--data_dir", type=str, default="MLEndSND_Public")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = files.find_files(data_dir, ext="wav")
    annots = pd.read_csv(data_dir / "MLEndSND_Audio_Attributes.csv", dtype=str)
    annots.set_index("Public filename", inplace=True)
    spk_info = pd.read_csv(data_dir / "MLEndSND_Speakers_Demographics.csv", dtype=str)
    spk_info.set_index("Speaker", inplace=True)
    df = annots.join(spk_info, "Speaker")

    # rename columns
    df.rename(
        columns={
            "Numeral": "numeral",
            "Intonation": "emotion",
            "Speaker": "speaker",
            "Language1": "language1",
            "Language2": "language2",
            "Nationality": "nationality",
            "Coordinates": "coordinates",
        },
        inplace=True,
    )

    df.index.name = "file"

    # add extension .wav to file
    df.index = df.index + ".wav"
    # allocate speaker > 200 for test
    train_df = df[df["speaker"].astype(int) <= 120]
    test_df = df.drop(train_df.index)

    # save to csv
    df.to_csv(output_dir / "mlendsnd.csv")
    train_df.to_csv(output_dir / "mlendsnd_train.csv")
    test_df.to_csv(output_dir / "mlendsnd_test.csv")

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
