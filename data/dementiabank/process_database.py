# process_database.py for DementiaBank dataset
# Following TensorFlow datasets split strategy with speaker-aware grouping

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd


def _compute_split_boundaries(split_probs, n_items):
    """Computes boundary indices for each of the splits in split_probs.

    Args:
        split_probs: List of (split_name, prob), e.g. [('train', 0.7), ('validation', 0.1), ('test', 0.2)]
        n_items: Number of items we want to split.

    Returns:
        The item indices of boundaries between different splits. For the above
        example and n_items=100, these will be
        [('train', 0, 70), ('validation', 70, 80), ('test', 80, 100)].
    """
    if len(split_probs) > n_items:
        raise ValueError(
            f"Not enough items for the splits. There are {len(split_probs)} "
            f"splits while there are only {n_items} items"
        )
    total_probs = sum(p for name, p in split_probs)
    if abs(1 - total_probs) > 1e-8:
        raise ValueError(f"Probs should sum up to 1. probs={split_probs}")

    split_boundaries = []
    sum_p = 0.0
    for name, p in split_probs:
        prev = sum_p
        sum_p += p
        split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

    # Guard against rounding errors.
    split_boundaries[-1] = (
        split_boundaries[-1][0],
        split_boundaries[-1][1],
        n_items,
    )

    return split_boundaries


def _get_inter_splits_by_group(items_and_groups, split_probs, split_number):
    """Split items to train/dev/test, so all items in group go into same split.

    Each group contains all the samples from the same speaker ID. The samples are
    splitted so that all each speaker belongs to exactly one split.

    Args:
        items_and_groups: Sequence of (item_id, group_id) pairs.
        split_probs: List of (split_name, prob), e.g. [('train', 0.7), ('validation', 0.1), ('test', 0.2)]
        split_number: Generated splits should change with split_number.

    Returns:
        Dictionary that looks like {split name -> list(ids)}.
    """
    groups = sorted(set(group_id for item_id, group_id in items_and_groups))
    rng = np.random.RandomState(split_number)
    rng.shuffle(groups)

    split_boundaries = _compute_split_boundaries(split_probs, len(groups))
    group_id_to_split = {}
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            group_id_to_split[groups[i]] = split_name

    split_to_ids = collections.defaultdict(list)
    for item_id, group_id in items_and_groups:
        split = group_id_to_split[group_id]
        split_to_ids[split].append(item_id)

    return split_to_ids


def process_database(data_dir=None, output_dir=None):
    """
    Process the DementiaBank dataset to create train/val/test splits using TensorFlow datasets strategy.

    This follows the same speaker-aware splitting strategy as TensorFlow datasets:
    - 70% train, 10% validation, 20% test
    - Speaker-aware: all files from the same speaker go to the same split
    - Uses random state 0 for reproducibility (same as TF datasets)

    Args:
        data_dir: Path to the dementiabank directory (optional, defaults to current directory)
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

    print(f"Processing DementiaBank dataset from {data_dir}")

    # Define paths following TensorFlow datasets structure
    control_folder = data_dir / "DEMENTIABANK/English/Pitt/Control/cookie"
    dementia_folder = data_dir / "DEMENTIABANK/English/Pitt/Dementia/cookie"

    # Check if directories exist
    if not control_folder.exists():
        raise FileNotFoundError(f"Control folder not found: {control_folder}")
    if not dementia_folder.exists():
        raise FileNotFoundError(f"Dementia folder not found: {dementia_folder}")

    # Collect examples and speaker IDs (following TF datasets pattern)
    examples_and_speaker_ids = []

    # Process control files
    for mp3_file in control_folder.glob("*.mp3"):
        filename = mp3_file.name
        # Extract speaker ID from filename (e.g., "002-0.mp3" -> "002")
        speaker_id = filename.split("-")[0]

        # Create relative path for nkululeko
        relative_path = f"{control_folder}/{filename}"

        example = {"file": relative_path, "label": "control", "speaker_id": speaker_id}
        examples_and_speaker_ids.append((example, speaker_id))

    # Process dementia files
    for mp3_file in dementia_folder.glob("*.mp3"):
        filename = mp3_file.name
        # Extract speaker ID from filename (e.g., "001-0.mp3" -> "001")
        speaker_id = filename.split("-")[0]

        # Create relative path for nkululeko
        relative_path = f"{dementia_folder}/{filename}"

        example = {"file": relative_path, "label": "dementia", "speaker_id": speaker_id}
        examples_and_speaker_ids.append((example, speaker_id))

    print(f"Found {len(examples_and_speaker_ids)} total files")

    # Count unique speakers and class distribution
    unique_speakers = set(speaker_id for _, speaker_id in examples_and_speaker_ids)
    control_files = [
        ex for ex, _ in examples_and_speaker_ids if ex["label"] == "control"
    ]
    dementia_files = [
        ex for ex, _ in examples_and_speaker_ids if ex["label"] == "dementia"
    ]

    print(f"Control files: {len(control_files)}")
    print(f"Dementia files: {len(dementia_files)}")
    print(f"Unique speakers: {len(unique_speakers)}")

    # Use TensorFlow datasets split strategy: 70% train, 10% validation, 20% test
    split_probs = [("train", 0.7), ("validation", 0.1), ("test", 0.2)]

    # Apply speaker-aware splitting (same as TF datasets with split_number=0)
    splits = _get_inter_splits_by_group(examples_and_speaker_ids, split_probs, 0)

    # Create DataFrames for each split
    split_dfs = {}
    for split_name in ["train", "validation", "test"]:
        examples = splits[split_name]

        # Create DataFrame
        data = []
        for example in examples:
            data.append(
                {
                    "file": example["file"],
                    "dementia": example["label"],
                    "speaker": f"DEMENTIABANK_{example['speaker_id']}",
                    "gender": "unknown",  # DementiaBank doesn't include gender info
                }
            )

        df = pd.DataFrame(data)
        df = df.set_index("file")
        split_dfs[split_name] = df

    # Save the splits
    output_dir = Path(output_dir)
    split_dfs["train"].to_csv(output_dir / "dementiabank_train.csv")
    split_dfs["validation"].to_csv(output_dir / "dementiabank_val.csv")
    split_dfs["test"].to_csv(output_dir / "dementiabank_test.csv")

    # Also create a combined file for convenience
    df_all = pd.concat([split_dfs["train"], split_dfs["validation"], split_dfs["test"]])
    df_all.to_csv(output_dir / "dementiabank.csv")

    # Print statistics
    print("\nDataset statistics (following TensorFlow datasets split):")
    print(f"Train samples: {len(split_dfs['train'])}")
    print(f"  - dementia: {sum(split_dfs['train']['dementia'] == 'dementia')}")
    print(f"  - control: {sum(split_dfs['train']['dementia'] == 'control')}")

    print(f"Validation samples: {len(split_dfs['validation'])}")
    print(f"  - dementia: {sum(split_dfs['validation']['dementia'] == 'dementia')}")
    print(f"  - control: {sum(split_dfs['validation']['dementia'] == 'control')}")

    print(f"Test samples: {len(split_dfs['test'])}")
    print(f"  - dementia: {sum(split_dfs['test']['dementia'] == 'dementia')}")
    print(f"  - control: {sum(split_dfs['test']['dementia'] == 'control')}")

    print(f"Total samples: {len(df_all)}")
    print(f"Unique speakers: {df_all['speaker'].nunique()}")

    # Verify speaker separation
    train_speakers = set(split_dfs["train"]["speaker"])
    val_speakers = set(split_dfs["validation"]["speaker"])
    test_speakers = set(split_dfs["test"]["speaker"])

    overlap_train_val = train_speakers & val_speakers
    overlap_train_test = train_speakers & test_speakers
    overlap_val_test = val_speakers & test_speakers

    print("\nSpeaker separation verification:")
    print(f"Train speakers: {len(train_speakers)}")
    print(f"Validation speakers: {len(val_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")
    print(f"Train-Val overlap: {len(overlap_train_val)} (should be 0)")
    print(f"Train-Test overlap: {len(overlap_train_test)} (should be 0)")
    print(f"Val-Test overlap: {len(overlap_val_test)} (should be 0)")

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("WARNING: Speaker overlap detected!")
    else:
        print("✓ Perfect speaker separation achieved")

    print(f"\nFiles saved to {output_dir}:")
    print("- dementiabank_train.csv")
    print("- dementiabank_val.csv")
    print("- dementiabank_test.csv")
    print("- dementiabank.csv (combined)")


def main():
    parser = argparse.ArgumentParser(
        description="Process DementiaBank dataset for nkululeko using TensorFlow datasets split strategy"
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default=".",
        help="Path to dementiabank directory containing DEMENTIABANK folder",
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
