#!/usr/bin/env python3
"""
Process the Fake-or-Real (FoR) 2-second dataset and generate CSV files.

The FoR-2sec dataset contains synthetic and real speech samples organized into
training, validation, and testing sets with fake/real subdirectories.

Dataset structure:
for-2seconds/
├── training/
│   ├── fake/
│   └── real/
├── validation/
│   ├── fake/
│   └── real/
└── testing/
    ├── fake/
    └── real/

Output CSV files:
- for-2sec_train.csv
- for-2sec_dev.csv (validation set)
- for-2sec_test.csv
- for-2sec.csv (all combined)

Columns: file, label
- file: relative path to the audio file
- label: 'fake' or 'real'
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add nkululeko parent directory to path to enable imports
script_dir = Path(__file__).parent
nkululeko_root = script_dir.parent.parent
sys.path.insert(0, str(nkululeko_root))

try:
    from nkululeko.utils.files import find_files
except ImportError:
    # Fallback to glob if nkululeko is not available
    def find_files(directory, ext=None, relative=False):
        """Find all files with given extensions in directory."""
        directory = Path(directory)
        if ext is None:
            ext = ["*"]
        files = []
        for extension in ext:
            files.extend(directory.glob(f"*.{extension}"))
        return sorted(files)


def process_database(data_dir, output_dir):
    """
    Process the FoR-2sec dataset and generate CSV files.
    
    Args:
        data_dir: Path to the for-2seconds directory
        output_dir: Path to save the CSV files
    """
    # Check if data_dir exists
    data_dir = Path(data_dir).resolve()  # Convert to absolute path
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ERROR: Directory not found: {data_dir}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir).resolve()  # Convert to absolute path
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Process each split (train, validation, test)
    splits = {
        "train": "training",
        "dev": "validation",
        "test": "testing"
    }
    
    all_data = []
    
    for split_name, split_dir in splits.items():
        split_path = data_dir / split_dir
        
        if not split_path.exists():
            print(f"WARNING: Split directory not found: {split_path}")
            continue
        
        # Process fake and real subdirectories
        fake_files = find_files(split_path / "fake", ext=["wav"], relative=False, path_object=True)
        real_files = find_files(split_path / "real", ext=["wav"], relative=False, path_object=True)
        
        # Create dataframe for this split
        fake_df = pd.DataFrame({
            "file": [str(Path(f).relative_to(data_dir.parent)) for f in fake_files],
            "label": ["fake"] * len(fake_files)
        })
        
        real_df = pd.DataFrame({
            "file": [str(Path(f).relative_to(data_dir.parent)) for f in real_files],
            "label": ["real"] * len(real_files)
        })
        
        # Combine fake and real
        split_df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Save split CSV
        csv_path = output_dir / f"for-2sec_{split_name}.csv"
        split_df.to_csv(csv_path, index=False)
        
        print(f"✓ Saved {len(split_df)} samples to {csv_path}")
        print(f"  - Fake: {len(fake_df)}, Real: {len(real_df)}")
        
        # Add to all_data for combined CSV
        all_data.append(split_df)
    
    # Create combined CSV with all splits
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv = output_dir / "for-2sec.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✓ Saved {len(combined_df)} total samples to {combined_csv}")
        print(f"  - Fake: {len(combined_df[combined_df['label'] == 'fake'])}")
        print(f"  - Real: {len(combined_df[combined_df['label'] == 'real'])}")
    
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process FoR-2sec dataset and generate CSV files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./for-2seconds/",
        help="Path to the for-2seconds directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to the output directory for CSV files"
    )
    args = parser.parse_args()
    
    process_database(args.data_dir, args.output_dir)
