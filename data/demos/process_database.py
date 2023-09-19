#!/usr/bin/env python3
# process_database.py --> DEMoS

import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "rab": "anger",
    "tri": "sadness",
    "gio": "happiness",
    "pau": "fear",
    "dis": "disgust",
    "col": "guilt",
    "sor": "surprise",
    "neu": "neutral",
}

gender_map = {
    "f": "female",
    "m": "male"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='DEMoS', help='Path to the database file')
    parser.add_argument('--output_dir', type=str, default='.', help='Path to the output directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()

    paths = list(data_dir.glob('**/*.wav'))
    # emotions = [emotion_map[p.stem.split('_')[3][:3]] if p.parts[1] == 'DEMOS' else emotion_map[p.stem.split('_')[2][:3]] for p in paths]

    emotions = [emotion_map[p.stem.split('_')[-1][:3]] for p in paths]
    genders = [gender_map[p.stem.split('_')[-3]] for p in paths]
    speakers = [p.stem.split('_')[-2] for p in paths]
    languages = ["italian" for p in paths]
    prototypicality = [p.stem[:2] if p.stem[:2] in {"NP", "PR"} else "neutral" for p in paths]


    # convert to df
    df = pd.DataFrame({"file": paths, "emotion": emotions, "gender": genders, "speaker": speakers, "language": languages, "prototypicality": prototypicality})

    # split train and test based in speaker independent and balanced emotion
    # allocate speakers >= 55 for test
    df_test = df[df["speaker"] > "57"]  #python3.9 newer 
    # df_test = df[df["speaker"].astype(int) > 57]
    df_train = df.drop(df_test.index)

    # save to csv
    df.to_csv(output_dir / "demos.csv", index=False)
    df_train.to_csv(output_dir / "demos_train.csv", index=False)
    df_test.to_csv(output_dir / "demos_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")


if __name__ == "__main__":
    main()

