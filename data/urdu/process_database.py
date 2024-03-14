#!/usr/bin/env python3
# process_database.py --> URDU

import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "A": "anger",
    "S": "sadness",
    "H": "happiness",
    "N": "neutral",
}


gender_map = {
    "M": "male",
    "F": "female"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='URDU', help='Path to the database file')
    parser.add_argument('--output_dir', type=str, default='.', help='Path to the output directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()

    paths = list(data_dir.glob('**/*.wav'))
    emotions = [emotion_map[p.stem[p.stem.rfind("_") + 1]] for p in paths]
    speakers = [p.stem[: p.stem.index("_")] for p in paths]
    genders = [gender_map[s[1]] for s in speakers]
    languages = ["urdu" for p in paths]

    df = pd.DataFrame({"file": paths, "emotion": emotions, "speaker": speakers, "gender": genders, "language": languages})


    # allocate speaker independent train/test --> very low result
    # df_test = df[df["speaker"].isin(["SM3", "SM5", "SF10", "SF9", "SF11"])]
    # df_train = df.drop(df_test.index)

    # use train_test_split
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["emotion"])


    # save to csv
    df.to_csv(output_dir / "urdu.csv", index=False)
    df_train.to_csv(output_dir / "urdu_train.csv", index=False)
    df_test.to_csv(output_dir / "urdu_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")


if __name__ == "__main__":
    main()
