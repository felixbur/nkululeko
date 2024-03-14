# process_database.ini --> subesco
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/SUBESCO/process.py
import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

REGEX = re.compile(r"^([MF])_([0-9]+)_([A-Z]+)_S_([0-9]+)_([A-Z]+)_([0-9])$")

emotion_map = {
    "ANGRY": "anger",
    "DISGUST": "disgust",
    "HAPPY": "happiness",
    "NEUTRAL": "neutral",
    "FEAR": "fear",
    "SURPRISE": "surprise",
    "SAD": "sadness",
}

gender_map = {
    'M': "male",
    'F': "female"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="SUBESCO")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load data
    wav_files = list(data_dir.glob("**/*.wav"))
    keys = ["file", "emotion", "speaker", "gender", "language"]
    annot: Dict[str, Dict[str, str]] = {x: {} for x in keys}

    file = [wav_file for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]
    emotion = [emotion_map[REGEX.match(wav_file.stem)[5]] for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]
    speaker_idx = [REGEX.match(wav_file.stem)[2] for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]
    speaker = [REGEX.match(wav_file.stem)[3] for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]
    gender = [gender_map[REGEX.match(wav_file.stem)[1]] for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]
    language = ["bangla" for wav_file in wav_files if REGEX.match(wav_file.stem) is not None]

    # for wav_file in wav_files:
    #     match = REGEX.match(wav_files.stem)
    #     if match is None:
    #         raise ValueError(f"Invalid filename: {wav_file.stem}")
    #     else:
    #         annot["file"][wav_file.stem] = wav_file
    #         annot["emotion"][wav_file.stem] = emotion_map[match[5]]
    #         annot["speaker"][wav_file.stem] = match[2]
    #         annot["gender"][wav_file.stem] = match[1]
    #     # language is same for all files
    #     # annot["language"][wav_file.stem] = "bangla"

    # # collect all data
    # annot = {k: list(v.values()) for k, v in annot.items()} 
    
    # Create dataframe
    df = pd.DataFrame({"file": file, "emotion": emotion, "gender": gender, 
                       "speaker": speaker, "speaker_idx": speaker_idx, 
                       "language": language})
    
    # allocate speaker 9 and 10 from each male and female gender to test set
    test_df = df[df["speaker_idx"].isin(["09", "10"])]
    train_df = df.drop(test_df.index)
    train_df.to_csv('subesco_train.csv', index=False)
    test_df.to_csv('subesco_test.csv', index=False)

    print(f"Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")
    
if __name__ == "__main__":
    main()
