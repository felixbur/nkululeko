#!/usr/bin/env python3
# process_database.py --> EMORYNLP

import argparse
from pathlib import Path

import pandas as pd

NAME_FMT = "sea{0.Season}_ep{0.Episode}_sc{0.Scene_ID}_utt{0.Utterance_ID}"

emotion_map = {
    "Joyful": "happiness",
    "Mad": "anger",
    "Neutral": "neutral",
    "Peaceful": "peace",
    "Powerful": "power",
    "Sad": "sadness",
    "Scared": "fear",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="EMORYNLP")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    splits = ["train", "dev", "test"]

    for split in splits:
        input_csv_file = data_dir / f"emorynlp_{split}_final.csv"
        df = pd.read_csv(input_csv_file)
        df.index = df.apply(lambda x: NAME_FMT.format(x), axis=1)
        df["split"] = split
        if split == "train":
            df = df.drop(
                ["sea4_ep4_sc3_utt10", "sea4_ep4_sc3_utt8", "sea4_ep4_sc3_utt9"])

        df["Emotion"] = df["Emotion"].map(emotion_map)
        df = df.rename(columns={"Emotion": "emotion",
                       "Sentiment": "sentiment", "Speaker": "speaker"})
        df.index = df.index + ".wav"
        df.index.name = "file"
        df.to_csv(output_dir / f"emorynlp_{split}.csv")
        df.index = df.index.str.replace(".wav", ".mp4")
        df.to_csv(output_dir / f"emorynlp_{split}_mp4.csv")

        # print number of each split
        print(f"{split}: {len(df)}")


if __name__ == "__main__":
    main()
