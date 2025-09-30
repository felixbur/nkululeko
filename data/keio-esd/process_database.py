# process_database.py: for processing various datasets

import argparse
import csv
import os
import sys

from pathlib import Path

# Add the nkululeko parent directory to Python path
nkululeko_path = Path(__file__).parents[2]  
sys.path.insert(0, str(nkululeko_path))

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):
    # Define the mapping of folders to sets
    set_mapping = {
        "airuraNdo": "train",
        "aomori": "train",
        "arakajime": "train",
        "baieruN": "train",
        "burukkuriN": "train",
        "erabu": "train",
        "iNguraNdo": "train",
        "kagawa": "dev",
        "kumamoto": "dev",
        "mie": "test",
        "mizu": "test",
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
        parts = basename.split("_")
        if len(parts) < 3:
            print(f"Skipping file with unexpected format: {basename}")
            continue
        emotion = parts[1]  # Extract emotion
        intensity = parts[2][0]  # Extract intensity

        # Use set_mapping to determine set name
        sub_dir = os.path.basename(os.path.dirname(wav))
        set_name = set_mapping.get(sub_dir, None)
        if set_name is None:
            print(f"Skipping file with unknown set mapping: {wav}")
            continue

        # get relative path only
        full_path = wav
        # full_path = wav.replace("C:\\Users\\Atoz_\\nkululeko\\data\\Keio-ESD\\", "")

        # Append to the corresponding set with cleaned path
        data[set_name].append([full_path, "male", emotion, intensity])

    # Write the data to CSV files for each set
    # get human or synthesized from data_dir
    speech_type = "human" if "human" in data_dir else "synthesized"
    for set_name, set_data in data.items():
        csv_path = os.path.join(output_dir, f"keio_{speech_type}_{set_name}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file", "gender", "emotion", "intensity"])
            writer.writerows(set_data)
        print(f"Wrote {len(set_data)} entries to {csv_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="KEIO-ESD/synthesized",  # human or synthesized
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