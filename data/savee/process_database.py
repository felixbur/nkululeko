# process_databse.py --> SAVEE
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/SAVEE/process.py

import argparse
import re
from pathlib import Path

import pandas as pd

REGEX = re.compile(r"^([a-z][a-z]?)[0-9][0-9]$")

emotion_map = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="SAVEE", help="Path of SAVEE directory (AudioData renamed to SAVEE)")
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    wav_files = list(data_dir.glob("**/*.wav"))
    file = [wav_file for wav_file in wav_files]
    speaker = [wav_file.parts[1] for wav_file in wav_files]
    emotion = [emotion_map[REGEX.match(wav_file.stem).group(1)] for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]
    gender = ["male" for wav_file in wav_files]
    language = ["english" for wav_file in wav_files]
    
    # convert to df
    df = pd.DataFrame({"file": file, "speaker": speaker, "emotion": emotion, 
        "language": language, "gender": gender})

    # allocate the last speaker, KL, for test set
    test_df = df[df["speaker"] == "KL"]
    train_df = df.drop(test_df.index)

    # save to CSV
    train_df.to_csv(out_dir / "savee_train.csv", index=False)
    test_df.to_csv(out_dir / "savee_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
if __name__ == "__main__":
    main()
