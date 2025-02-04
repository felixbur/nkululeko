# process_database.py: for JVNV database
# export PYTHONPATH="${PYTHONPATH}:/home/bagus/github/nkululeko"

import argparse
import csv
import os

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):
    # Define the mapping of folders to sets
    set_mapping = {
        "F1": "train",
        "F2": "dev",
        "M1": "train",
        "M2": "test",
    }

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to store the data for each set
    data = {"train": [], "dev": [], "test": []}

    # Find all wav files using Nkululeko find tools
    wavs = find_files(data_dir, relative=True, ext=["wav"])

    # get basename
    for wav in wavs:
        basename = os.path.basename(wav)
        # get emotion
        emotion = basename.split("_")[1]
        # get gender
        gender = basename.split("_")[0][0]
        # map gender to string, M to male, F to female
        gender = "male" if gender == "M" else "female"
        # get set name, M1 and F1 for train, M2 for dev, F2 for test
        set_name = set_mapping[basename.split("_")[0]]
        # append to data
        data[set_name].append([wav, emotion, gender])

    # Write the data to CSV files for each set
    for set_name, set_data in data.items():
        csv_path = os.path.join(output_dir, f"jvnv_{set_name}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file", "emotion", "gender"])
            writer.writerows(set_data)

    # print number of files in each set
    for set_name, set_data in data.items():
        print(f"Number of files in {set_name} set: {len(set_data)}")
    print("Database processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JVNV database")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="jvnv_v1",
        help="Directory containing the JVNV data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to store the output CSV files",
    )
    args = parser.parse_args()

    process_database(args.data_dir, args.output_dir)
# process_database.py: pre-processing script for JNVV database
