# process_database.py: pre-processing script for Nemo database


# from nkululeko.utils import find_files
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def process_database(data_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # read data.tsv
    data_path = os.path.join(data_dir, "data.tsv")
    df = pd.read_csv(data_path, sep="\t")

    # rename column file_id to file, speaker_id to speaker
    df.rename(columns={"file_id": "file",
              "speaker_id": "speaker"}, inplace=True)

    # Split the data into train, dev, and test sets based speaker independently
    speaker_ids = df['speaker'].unique()
    train_speaker_ids, temp_speaker_ids = train_test_split(
        speaker_ids, test_size=0.3, random_state=42)
    dev_speaker_ids, test_speaker_ids = train_test_split(
        temp_speaker_ids, test_size=0.5, random_state=42)

    train_df = df[df['speaker'].isin(train_speaker_ids)]
    dev_df = df[df['speaker'].isin(dev_speaker_ids)]
    test_df = df[df['speaker'].isin(test_speaker_ids)]

    # train_df, temp_df = train_test_split(
    #     df, test_size=0.3, stratify=df['speaker'], random_state=42)
    # dev_df, test_df = train_test_split(
    #     temp_df, test_size=0.5, stratify=temp_df['speaker'], random_state=42)

    # Write the data to CSV files for each set
    train_df.to_csv(os.path.join(output_dir, "nemo_train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "nemo_dev.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "nemo_test.csv"), index=False)
    # print number of train, dev, and test samples
    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of dev samples: {len(dev_df)}")
    print(f"Number of test samples: {len(test_df)}")

    print("Database processing completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Nemo database")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="nEMO",
        help="Directory containing the Nemo data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to store the output CSV files",
    )
    args = parser.parse_args()

    process_database(args.data_dir, args.output_dir)
