#!/usr/bin/env python3
"""
Process the E-SERAVD dataset and generate CSV files.

E-SERAVD is an Indonesian Speech Emotion Recognition (SER) corpus developed by
the Telecommunication Signal Processing Laboratory, EEPIS. It consists of
emotional speech segments extracted from Indonesian film dialogues.

File naming convention: <prefix>-<emotion>-<speaker>-<status>-<number>.wav
- Prefix: 02 (dataset version)
- Emotion: 01=angry, 02=sad, 03=neutral, 04=happy
- Speaker: AADC, D1990, DHB, HA3, HSL, TAOL (6 speakers)
- Status: T (Tested/validated)
- Number: 0001-XXXX (sample number)

Directory structure:
E-SERAVD/02_Audio-Only/<emotion>/<speaker>/<filename>.wav

Output CSV files:
- e-seravd_train.csv
- e-seravd_dev.csv
- e-seravd_test.csv
- e-seravd.csv (all combined)

Columns: file, speaker, emotion, language, status
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add nkululeko parent directory to path to enable imports
script_dir = Path(__file__).parent
nkululeko_root = script_dir.parent.parent
sys.path.insert(0, str(nkululeko_root))

try:
    from nkululeko.utils.files import find_files
except ImportError:
    # Fallback to glob if nkululeko is not available
    def find_files(directory, ext=None, relative=False, path_object=False):
        """Find all files with given extensions in directory."""
        directory = Path(directory)
        if ext is None:
            ext = ["*"]
        files = []
        for extension in ext:
            files.extend(directory.rglob(f"*.{extension}"))
        if path_object:
            return sorted(files)
        else:
            return sorted([str(f) for f in files])


# Emotion mapping from directory names and file codes
EMOTION_MAP = {
    "01": "angry",
    "02": "sad",
    "03": "neutral",
    "04": "happy"
}


def process_database(data_dir, output_dir):
    """
    Process the E-SERAVD dataset and generate CSV files.
    
    Args:
        data_dir: Path to the E-SERAVD/02_Audio-Only directory
        output_dir: Path to save the CSV files
    """
    # Check if data_dir exists
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ERROR: Directory not found: {data_dir}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Find all wav files
    wav_files = find_files(data_dir, ext=["wav"], relative=False, path_object=True)
    print(f"Found {len(wav_files)} wav files.")
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No wav files found in {data_dir}")
    
    # Parse filename components
    data = []
    for wav_file in wav_files:
        # Get relative path from data_dir.parent for CSV
        rel_path = wav_file.relative_to(data_dir.parent)
        
        # Parse filename: 02-<emotion>-<speaker>-<status>-<number>.wav
        filename = wav_file.stem
        parts = filename.split("-")
        
        if len(parts) != 5:
            print(f"Warning: Skipping file with unexpected format: {filename}")
            continue
        
        prefix, emotion_code, speaker, status, number = parts
        
        # Validate format
        if prefix != "02":
            print(f"Warning: Unexpected prefix '{prefix}' in {filename}")
            continue
        
        if status != "T":
            print(f"Warning: Skipping non-tested file: {filename}")
            continue
        
        # Map emotion
        emotion = EMOTION_MAP.get(emotion_code, f"unknown_{emotion_code}")
        
        data.append({
            "file": str(rel_path),
            "speaker": speaker,
            "emotion": emotion,
            "language": "indonesian",
            "status": status
        })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError("No valid audio files found to process")
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Speakers: {df['speaker'].nunique()}")
    print(f"Speaker list: {sorted(df['speaker'].unique())}")
    print(f"Emotions: {df['emotion'].value_counts().to_dict()}")
    
    # Speaker-independent split (train: 60%, dev: 20%, test: 20%)
    speakers = df["speaker"].unique()
    print(f"\nTotal speakers: {len(speakers)}")
    
    train_speakers, temp_speakers = train_test_split(
        speakers, test_size=0.4, random_state=42
    )
    dev_speakers, test_speakers = train_test_split(
        temp_speakers, test_size=0.5, random_state=42
    )
    
    # Create train, dev, test splits
    df_train = df[df["speaker"].isin(train_speakers)]
    df_dev = df[df["speaker"].isin(dev_speakers)]
    df_test = df[df["speaker"].isin(test_speakers)]
    
    # Save CSV files
    csv_files = {
        "train": df_train,
        "dev": df_dev,
        "test": df_test
    }
    
    for split_name, split_df in csv_files.items():
        csv_path = output_dir / f"e-seravd_{split_name}.csv"
        split_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved {len(split_df)} samples to {csv_path}")
        print(f"  Speakers: {sorted(split_df['speaker'].unique())}")
        print(f"  Emotions: {split_df['emotion'].value_counts().to_dict()}")
    
    # Save combined CSV
    combined_csv = output_dir / "e-seravd.csv"
    df.to_csv(combined_csv, index=False)
    print(f"\n✓ Saved {len(df)} total samples to {combined_csv}")
    
    print("\n✅ DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process E-SERAVD dataset and generate CSV files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./E-SERAVD/02_Audio-Only",
        help="Path to the E-SERAVD/02_Audio-Only directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path to the output directory for CSV files"
    )
    args = parser.parse_args()
    
    process_database(args.data_dir, args.output_dir)
