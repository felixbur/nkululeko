# process_database.py: for processing various datasets

import argparse
import csv
import os
import sys
from pathlib import Path

# Add the nkululeko parent directory to Python path
nkululeko_path = Path(__file__).parents[2]  # Go up two levels from data/ogvc to nkululeko root
sys.path.insert(0, str(nkululeko_path))

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

    # Process each wav file
    for wav in wavs:
        basename = os.path.basename(wav)
        # Extract metadata from the filename
        gender = "male" if basename[0] == "M" else "female"  # 1st character
        emotion = basename[7:10]  # 8th to 10th characters
        intensity = basename[10]  # 11th character
        # Get parent directory name to determine set assignment
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(wav)))
        set_name = set_mapping[parent_dir]  # Reference set_mapping

        # Append to the corresponding set with full path
        data[set_name].append([wav, gender, emotion, intensity])

    # Write the data to CSV files for each set
    for set_name, set_data in data.items():
        csv_path = os.path.join(output_dir, f"ogvc_{set_name}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file", "gender", "emotion", "intensity"])
            writer.writerows(set_data)

    # Print number of files in each set
    for set_name, set_data in data.items():
        print(f"Number of files in {set_name} set: {len(set_data)}")
    print("Database processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="OGVC/Acted/wav/",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to store the output CSV files",
    )
    args = parser.parse_args()

    process_database(args.data_dir, args.output_dir)