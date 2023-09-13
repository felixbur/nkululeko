# process_database.py --> EMOVO
# modified from: https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/EMOVO/process.py

import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "dis": "disgust",
    "gio": "happiness",
    "neu": "neutral",
    "pau": "fear",
    "rab": "anger",
    "sor": "surprise",
    "tri": "sadness",
}

gender_map = {
    "f": "female",
    "m": "male"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="EMOVO")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob("**/*.wav"))
    files = [file for file in paths]
    emotion = [emotion_map[file.stem[:3]] for file in files]
    speaker = [file.stem[4:6] for file in files]
    gender = [gender_map[file.stem[4]] for file in files]
    language = ["italian" for file in files]

    # convert to df
    df = pd.DataFrame({"file": files, "emotion": emotion, "speaker": speaker, 
            "gender": gender, "language": language})

    # allocate last speaker f3 for test
    train_df = df[df["speaker"] != "f3"]
    test_df = df.drop(train_df.index)

    # save to csv
    df.to_csv(output_dir / "emovo.csv", index=False)
    train_df.to_csv(output_dir / "emovo_train.csv", index=False)
    test_df.to_csv(output_dir / "emovo_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
    
if __name__ == "__main__":
    main()
