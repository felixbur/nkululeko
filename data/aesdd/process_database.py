import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "a": "anger",
    "d": "disgust",
    "h": "happiness",
    "f": "fear",
    "s": "sadness",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="AESDD", help="Path of AESDD directory")
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    paths = list(data_dir.glob("**/*.wav"))
    files = [file for file in paths  if file.stem != "s05 (3)"]
    names = [file.stem for file in files]
    emotion = [emotion_map[file.stem[0]] for file in files]
    speaker = [str(int(x[x.find("(") + 1: x.find(")")])) for x in names]
    gender = [["female", "male"][int(v) % 2] for v in speaker]
    language =["greek" for file in files]

    # convert to df
    df = pd.DataFrame({"file": files, "speaker": speaker, "emotion": emotion, "gender": gender})

    # print distribution per emotion
    # print(df.groupby("emotion").count()['file'])

    # allocate speaker 5 for test set
    train_df = df[df["speaker"] != "5"]
    test_df = df.drop(train_df.index)

    # save to CSV
    df.to_csv(out_dir / "aesdd.csv", index=False)
    train_df.to_csv(out_dir / "aesdd_train.csv", index=False)
    test_df.to_csv(out_dir / "aesdd_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
    
if __name__ == "__main__":
    main()
