"""process_database.py for BanglaSER dataset

Filename convention for the dataset:
AA-BB-CC-DD-EE-FF-GG.wav
AA = mode, 03: audio only
BB = script, 01: scripted
CC = emotion, 01 = Happy, 02 = Sad, 03 = Angry, 04 = Surprise, 05 = Neutral
DD = intensity, 01 = Normal, 02 = Strong
EE = text
01: It's twelve o'clock
02: I knew something like this would happen.
03: What kind of gift is this?
FF = Repetition, 01, 02, 03
GG = Actor ID, 01-34 (odd: male, even: female)
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):
    # check if data_dir exists
    if not Path(data_dir).is_dir():
        print(f"ERROR: no such directory {data_dir}")
        return
    # create output dir if not exist
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir()

    # emotion mapping
    emotion_map = {
        "01": "happy",
        "02": "sad",
        "03": "angry",
        "04": "surprise",
        "05": "neutral",
    }

    # intensity mapping
    intensity_map = {"01": "normal", "02": "strong"}

    # gender mapping, odd: male, even: female
    gender_map = {
        str(i).zfill(2): "male" if int(i) % 2 != 0 else "female" for i in range(1, 35)
    }

    # load the data
    wavs = find_files(
        Path(data_dir) / "t9h6p943xy-5" / "BanglaSER", ext=["wav"], relative=True
    )

    data = []

    # loop over wavs
    for wav in wavs:
        # get basename without extension
        fname = Path(wav).stem
        # get emotion label from emotion_map
        emotion = emotion_map[fname.split("-")[2]]
        # get intensity label from intensity_map
        intensity = intensity_map[fname.split("-")[3]]
        # get gender label from gender_map
        gender = gender_map[fname.split("-")[6]]
        # get speaker ID
        speaker = "banglaser_" + str(fname.split("-")[6])
        # get script ID
        text_id = fname.split("-")[4]
        # get language
        language = "bangla"

        # add to data
        data.append(
            {
                "file": wav,
                "emotion": emotion,
                "intensity": intensity,
                "gender": gender,
                "speaker": speaker,
                "text_id": text_id,
                "language": language,
            }
        )

    # convert to dataframe
    df = pd.DataFrame(data)
    # print(df.head) for debug

    train_speakers = df["speaker"].unique()

    train_speakers, test_speakers = train_test_split(
        train_speakers, test_size=0.3, random_state=42)

    test_speakers, dev_speakers = train_test_split(
        test_speakers, test_size=0.5, random_state=42)

    train_df = df[df["speaker"].isin(train_speakers)]
    dev_df = df[df["speaker"].isin(dev_speakers)]
    test_df = df[df["speaker"].isin(test_speakers)]

    # save dataframes to csv
    for set in ["train", "dev", "test"]:
        df = eval(f"{set}_df")
        df.to_csv(Path(output_dir) / f"banglaser_{set}.csv", index=False)
        print(f"Number of {set} samples: {len(df)}")
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BanglaSER database")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Directory containing the BanglaSER data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to store the output CSV files",
    )
    args = parser.parse_args()
    process_database(args.data_dir, args.output_dir)
