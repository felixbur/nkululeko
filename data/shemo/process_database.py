# process_databse.py --> SheMO

import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "A": "anger",
    "H": "happiness",
    "N": "neutral",
    "S": "sadness",
    "W": "surprise",
    "F": "fear",
}

gender_map = {
    "F": "female",
    "M": "male"
}

# unused_emotions = ["F"]  # unused in Keesing's Emotion Repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="ShEMO")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob("**/*.wav"))
    files = [file for file in paths]
    names = [file.stem for file in files]
    emotion = [emotion_map[name[3]] for name in names]
    speaker = [name[:3] for name in names]
    gender = [gender_map[n[0]] for n in names]
    language = ["persian" for file in files]

    # convert to df
    df = pd.DataFrame({"file": files, "emotion": emotion, "speaker": speaker, "gender": gender, "language": language})

    # allocate speaker M12, M25, F07, F24 for test
    test_df = df[(df["speaker"] == "M12") | (df["speaker"] == "M25") | (df["speaker"] == "F07") | (df["speaker"] == "F24")]
    train_df = df.drop(test_df.index)


    # save to csv
    df.to_csv(output_dir / "shemo.csv", index=False)
    train_df.to_csv(output_dir / "shemo_train.csv", index=False)
    test_df.to_csv(output_dir / "shemo_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
    
if __name__ == "__main__":
    main()
