# process_database.py for BanglaSER dataset
# Path: data/banglaser/process_database.py

"""
Filename convention:
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

import os

import pandas as pd

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):
    # check if data_dir exists
    if not os.path.isdir(data_dir):
        print(f"ERROR: no such directory {data_dir}")
        return
    # create output dir if not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # emotion mapping
    emotion_map = {
        "01": "happy",
        "02": "sad",
        "03": "angry",
        "04": "surprise",
        "05": "neutral",
    }

    # intensity mapping
    intensity_map = {
        "01": "normal",
        "02": "strong",
    }

    # gender mapping, odd: male, even: female
    gender_map = {
        str(i): 'male' if i %
        2 != 0 else 'female' for i in range(
            1, 36)}

    # load the data
    wavs = find_files(data_dir, ext=["wav"], relative=True)

    data = []

    # loop over wavs
    for wav in wavs:
        # get emotion label from emotion_map
        emotion = emotion_map[wav.split("-")[2]]
        # get intensity label from intensity_map
        intensity = intensity_map[wav.split("-")[3]]
        # get gender label from gender_map
        gender = gender_map[wav.split("-")[6]]
        # get speaker ID
        speaker = wav.split("-")[6]
        # get script ID
        text_id = wav.split("-")[4]
        # get language
        language = "bangla"

        # add to data
        data.append({
            "wav": wav,
            "emotion": emotion,
            "intensity": intensity,
            "gender": gender,
            "speaker": speaker,
            "text_id": text_id,
            "language": language,
        })

    # convert to dataframe
    df = pd.DataFrame(data)

    # split to train,dev,test speaker independently, 8010:10
