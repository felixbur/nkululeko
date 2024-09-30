# process_database.py for KIA dataset

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

    # Dictionary to store the DataFrames for each fold and partition
    fold_partitions = {}

    csv_directory = Path(data_dir) / "hi_kia" / "split"
    # Iterate over the CSV files
    for file_path in csv_directory.glob("*.csv"):
        # Extract the fold number from the filename
        fold_number = int(file_path.stem.split("_")[0][1])

        # Extract the partition from the filename
        partition = file_path.stem.split("_")[1]

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Create a key for the fold and partition combination
        key = (fold_number, partition)

        # Store the DataFrame in the fold_partitions dictionary
        fold_partitions[key] = df

    # Generate the output CSV files for each fold and partition
    for (fold_number, partition), df in fold_partitions.items():
        # Create the output filename
        output_filename = f"kia_{partition}_{fold_number}.csv"
        output_path = Path(output_dir) / output_filename

        # aggregate emotion from column angry,sad,happy,neutral which has val 1
        df["emotion"] = df[["angry", "sad", "happy", "neutral"]].idxmax(axis=1)
        df["language"] = "korean"

        # rename fname to file, only save file and emotion columns
        df = df[["fname", "emotion", "language"]].rename(
            columns={"fname": "file"})

        # append .wav extension to file
        df["file"] = df["file"].apply(lambda x: x + ".wav")

        # Save the DataFrame to a new CSV file
        df.to_csv(output_path, index=False)

        print(f"Created {output_filename} with {len(df)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KIA database")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Directory containing the KIA data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to store the output CSV files",
    )
    args = parser.parse_args()
    process_database(args.data_dir, args.output_dir)
