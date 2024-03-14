# process_database.py --> CaFE
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/CaFE/process.py

import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "C": "anger",
    "D": "disgust",
    "J": "happiness",
    "N": "neutral",
    "P": "fear",
    "S": "surprise",
    "T": "sadness",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="CaFE")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob("**/*.wav"))
    files = [file for file in paths]
    emotion = [emotion_map[file.stem[3]] for file in files]
    speaker = [file.stem[:2] for file in files]
    language = ["french" for file in files]
    country = ["canada" for file in files]
    gender = [["female", "male"][int(v) % 2] for v in speaker]
    
    # convert to df
    df = pd.DataFrame({"file": files, "speaker": speaker, "emotion": emotion, "language": language, "country": country})

    # allocate the last two speakers (11, 12) for test
    train_df = df[(df["speaker"] != "11") & (df["speaker"] != "12")]
    test_df = df.drop(train_df.index)

    # save to CSV
    df.to_csv(output_dir / "cafe.csv", index=False)
    train_df.to_csv(output_dir / "cafe_train.csv", index=False)
    test_df.to_csv(output_dir / "cafe_test.csv", index=False)


if __name__ == "__main__":
    main()
