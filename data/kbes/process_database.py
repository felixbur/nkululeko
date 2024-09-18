# process_database.py for KBES dataset

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):
    # check if data_dir exist
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # read all wav files
    wavs = find_files(data_dir, relative=True, ext=["wav"])
    print(f"Found {len(wavs)} wav files.")

    # map emotion: 1 = Neutral, 2 = Happy, 3 = Sad, 4 = Angry, 5 = Disgust
    emotion_mapping = {
        1: "neutral",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "disgust"
    }

    # map intensity, 1 = low, 2 = high
    intensity_mapping = {
        1: 'low',
        2: 'high'
    }

    # map gender 0 = female, 1 = male
    gender_mapping = {
        0: 'female',
        1: 'male'
    }

    data = []
    for wav in wavs:
        # get basename
        basename = os.path.basename(wav)
        # get emotion
        emotion = emotion_mapping[int(basename.split("-")[0])]
        # get intensity
        intensity = intensity_mapping[int(basename.split("-")[1])]
        # get gender
        gender = gender_mapping[int(basename.split("-")[2])]
        # add language
        language = "bengali"
        # add to data list
        data.append({
            "file": wav,
            "emotion": emotion,
            "gender": gender,
            "intensity": intensity,
            "language": language
        })

    # create dataframe from data
    df = pd.DataFrame(data)
    # split the data into train, dev, and test sets, balanced by emotion
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['emotion'], random_state=42)
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42)
    # write dataframes to csv
    train_df.to_csv(os.path.join(
        output_dir, "kbes_train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "kbes_dev.csv"), index=False)
    test_df.to_csv(os.path.join(
        output_dir, "kbes_test.csv"), index=False)
    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of dev samples: {len(dev_df)}")
    print(f"Number of test samples: {len(test_df)}")
    print("Database processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KBES dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="KUET Bangla Emotional Speech (KBES) Dataset",
        help="Directory containing the KBES data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to store the output CSV files",
    )
    args = parser.parse_args()

    process_database(args.data_dir, args.output_dir)
