import os
import pandas as pd
import argparse
from pathlib import Path


def main(data_dir):
    """Process ITW (In The Wild) deepfake detection dataset.

    The dataset contains audio files in release_in_the_wild/ directory with labels
    in meta.csv (file, speaker, label columns where label is 'spoof' or 'bona-fide').
    """
    # Define paths
    audio_dir = os.path.join(data_dir, "release_in_the_wild")
    meta_file = os.path.join(audio_dir, "meta.csv")

    # Read metadata
    print(f"Reading metadata from {meta_file}...")
    df = pd.read_csv(meta_file)

    # Add full path to audio files
    df["file"] = df["file"].apply(lambda x: os.path.join(audio_dir, x))

    # Map labels: 'bona-fide' -> 'real', 'spoof' -> 'fake'
    df["label"] = df["label"].map({"bona-fide": "real", "spoof": "fake"})

    # Rename columns to match nkululeko format
    df = df.rename(columns={"file": "file", "speaker": "speaker", "label": "label"})

    # Reorder columns
    df = df[["file", "speaker", "label"]]

    # Calculate split sizes (60% train, 20% dev, 20% test)
    # Shuffle with fixed seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df_shuffled)

    train_end = int(0.6 * total)
    dev_end = int(0.8 * total)

    # Split data
    df_train = df_shuffled.iloc[:train_end]
    df_dev = df_shuffled.iloc[train_end:dev_end]
    df_test = df_shuffled.iloc[dev_end:]

    # Save to CSV files
    train_file = os.path.join(data_dir, "itw_train.csv")
    dev_file = os.path.join(data_dir, "itw_dev.csv")
    test_file = os.path.join(data_dir, "itw_test.csv")
    all_file = os.path.join(data_dir, "itw.csv")

    df_train.to_csv(train_file, index=False)
    df_dev.to_csv(dev_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_shuffled.to_csv(all_file, index=False)

    print(f"✓ Created {train_file} with {len(df_train)} samples")
    print(f"✓ Created {dev_file} with {len(df_dev)} samples")
    print(f"✓ Created {test_file} with {len(df_test)} samples")
    print(f"✓ Created {all_file} with {len(df_shuffled)} samples")

    # Print label distribution
    print("\nLabel distribution in complete dataset:")
    print(df["label"].value_counts())

    print("\nLabel distribution in train set:")
    print(df_train["label"].value_counts())

    print("\nLabel distribution in dev set:")
    print(df_dev["label"].value_counts())

    print("\nLabel distribution in test set:")
    print(df_test["label"].value_counts())
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ITW (In The Wild) deepfake detection dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to the ITW dataset directory (default: script directory)",
    )
    args = parser.parse_args()

    main(args.data_dir)
