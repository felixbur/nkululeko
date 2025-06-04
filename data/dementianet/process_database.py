# process_database.py for DementiaNet dataset

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def process_database(data_dir=None, output_dir=None):
    """
    Process the DementiaNet dataset to create train/val/test splits.

    Args:
        data_dir: Path to the dementianet directory (optional, defaults to current directory)
        output_dir: Path to output directory (optional, defaults to current directory)
    """
    # Set default directories
    if data_dir is None:
        data_dir = Path(".")
    else:
        data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)

    # Check if required input files exist
    train_file = data_dir / "train_dm.csv"
    valid_file = data_dir / "valid_dm.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"Valid file not found: {valid_file}")

    print(f"Processing DementiaNet dataset from {data_dir}")

    # Read the original train and valid files
    df_train_orig = pd.read_csv(train_file, sep="\t")
    df_valid_orig = pd.read_csv(valid_file, sep="\t")

    print(f"Original train samples: {len(df_train_orig)}")
    print(f"Original valid samples: {len(df_valid_orig)}")

    # Apply path conversion with label information
    def convert_path_with_label(row):
        """Convert absolute path to relative path using label information"""
        path_str = row["path"]
        label = row["label"]

        # Extract the filename and person directory from the path
        parts = path_str.split("/")
        filename = parts[-1]

        # Find the person directory (usually second to last part)
        if len(parts) >= 2:
            person_dir = parts[-2]
            return f"DEMENTIANET/{label}/{person_dir}/{filename}"

        return path_str  # fallback to original if parsing fails

    # Apply path conversion using both path and label
    df_train_orig["file"] = df_train_orig.apply(convert_path_with_label, axis=1)
    df_valid_orig["file"] = df_valid_orig.apply(convert_path_with_label, axis=1)

    # Extract speaker information from file names
    def extract_speaker(file_path):
        """Extract speaker name from file path"""
        try:
            # Get the person directory name from the path
            parts = file_path.split("/")
            if len(parts) >= 3:
                return "DEMENTIANET_" + parts[2].replace(" ", "_")
            else:
                # Fallback: extract from filename
                filename = parts[-1].split(".")[0]
                # Remove the suffix numbers/letters to get base speaker name
                speaker_base = filename.split("_")[0]
                return "DEMENTIANET_" + speaker_base
        except Exception:
            return "DEMENTIANET_unknown"

    # Add speaker and gender columns
    df_train_orig["speaker"] = df_train_orig["file"].apply(extract_speaker)
    df_valid_orig["speaker"] = df_valid_orig["file"].apply(extract_speaker)

    # Add gender (unknown for this dataset)
    df_train_orig["gender"] = "unknown"
    df_valid_orig["gender"] = "unknown"

    # Rename 'label' column to 'dementia' for consistency
    df_train_orig = df_train_orig.rename(columns={"label": "dementia"})
    df_valid_orig = df_valid_orig.rename(columns={"label": "dementia"})

    # Select relevant columns and set index to file
    columns_to_keep = ["dementia", "speaker", "gender"]
    df_train_orig = df_train_orig.set_index("file")[columns_to_keep]
    df_valid_orig = df_valid_orig.set_index("file")[columns_to_keep]

    # Split the original training data: 90% train, 10% validation
    # Use stratified split to maintain class balance
    train_files = df_train_orig.index.tolist()
    train_labels = df_train_orig["dementia"].tolist()

    train_files_final, val_files, train_labels_final, val_labels = train_test_split(
        train_files, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # Create final splits
    df_train_final = df_train_orig.loc[train_files_final]
    df_val_final = df_train_orig.loc[val_files]
    df_test_final = df_valid_orig  # Use original valid as test

    # Save the splits
    output_dir = Path(output_dir)
    df_train_final.to_csv(output_dir / "dementianet_train.csv")
    df_val_final.to_csv(output_dir / "dementianet_val.csv")
    df_test_final.to_csv(output_dir / "dementianet_test.csv")

    # Also create a combined file for convenience
    df_all = pd.concat([df_train_final, df_val_final, df_test_final])
    df_all.to_csv(output_dir / "dementianet.csv")

    # Print statistics
    print("\nDataset statistics:")
    print(f"Train samples: {len(df_train_final)}")
    print(f"  - dementia: {sum(df_train_final['dementia'] == 'dementia')}")
    print(f"  - nodementia: {sum(df_train_final['dementia'] == 'nodementia')}")

    print(f"Validation samples: {len(df_val_final)}")
    print(f"  - dementia: {sum(df_val_final['dementia'] == 'dementia')}")
    print(f"  - nodementia: {sum(df_val_final['dementia'] == 'nodementia')}")

    print(f"Test samples: {len(df_test_final)}")
    print(f"  - dementia: {sum(df_test_final['dementia'] == 'dementia')}")
    print(f"  - nodementia: {sum(df_test_final['dementia'] == 'nodementia')}")

    print(f"Total samples: {len(df_all)}")
    print(f"Unique speakers: {df_all['speaker'].nunique()}")

    print(f"\nFiles saved to {output_dir}:")
    print("- dementianet_train.csv")
    print("- dementianet_val.csv")
    print("- dementianet_test.csv")
    print("- dementianet.csv (combined)")


def main():
    parser = argparse.ArgumentParser(
        description="Process DementiaNet dataset for nkululeko"
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default=".",
        help="Path to dementianet directory containing train_dm.csv and valid_dm.csv",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for processed CSV files (defaults to data_dir)",
    )

    args = parser.parse_args()

    try:
        process_database(args.data_dir, args.output_dir)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
