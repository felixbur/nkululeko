#!/usr/bin/env python3
"""
Pre-processing script for MELD-ST dataset to generate train/dev/test CSVs
compatible with nkululeko format.

This script processes the MELD-ST (Multimodal EmotionLines Dataset - Speech Translation)
dataset for both ENG_DEU and ENG_JPN language pairs, creating CSV files with the
required format: file, emotion, sentiment, speaker, language_pair, and transcription columns

The MELD-ST dataset contains parallel audio files for English and translated versions
(German/Japanese) with emotion, sentiment, speaker annotations, and transcriptions.

Generated CSVs will include:
- Standard nkululeko columns: file, emotion, sentiment, speaker, language, language_pair
- Text transcription columns:
  - text (in target language: English/German/Japanese)
  - English (original English transcription as reference)
"""

import pandas as pd
import os
import argparse
from pathlib import Path


def process_meld_st_split(
    csv_path, audio_base_path, language_pair, split_name, target_language
):
    """
    Process a single MELD-ST CSV file and generate nkululeko-compatible format for one language.

    Args:
        csv_path: Path to the MELD-ST CSV file
        audio_base_path: Base path to audio files
        language_pair: Either 'ENG_DEU' or 'ENG_JPN'
        split_name: 'train', 'valid', or 'test'
        target_language: Language to process ('ENG', 'DEU', or 'JPN')

    Returns:
        DataFrame with nkululeko format for the specified language
    """
    print(f"Processing {language_pair} {split_name} split for {target_language}...")

    # Read the original CSV
    df = pd.read_csv(csv_path)
    print(f"  Original data shape: {df.shape}")
    print(f"  Available columns: {list(df.columns)}")

    # Create output data for the specified language
    output_rows = []

    # Map split names to audio directory names
    split_map = {"train": "train", "valid": "dev", "test": "test"}
    audio_split = split_map[split_name]

    # Determine which column to use for text transcription based on target language
    language_to_column_map = {"ENG": "English", "DEU": "German", "JPN": "Japanese"}

    # Get the appropriate transcription column for target language
    target_text_col = language_to_column_map.get(target_language)
    english_text_col = "English"  # Always try to include English as reference

    print(
        f"  Target language: {target_language}, looking for column: {target_text_col}"
    )

    # Check if columns exist
    has_target_text = target_text_col and target_text_col in df.columns
    has_english_text = english_text_col in df.columns

    print(f"  Found {target_text_col}: {has_target_text}")
    print(f"  Found English: {has_english_text}")

    if has_target_text:
        sample_vals = df[target_text_col].head(2).tolist()
        print(f"  Sample {target_text_col} data: {sample_vals}")

    if has_english_text and target_language != "ENG":
        sample_vals = df[english_text_col].head(2).tolist()
        print(f"  Sample English data: {sample_vals}")

    for _, row in df.iterrows():
        # Process specified language
        audio_file = f"MELD-ST/{language_pair}/{target_language}/{audio_split}/{audio_split}_{row['id']}.wav"
        full_path = os.path.join(
            audio_base_path,
            f"{target_language}/{audio_split}/{audio_split}_{row['id']}.wav",
        )

        if os.path.exists(full_path):
            output_row = {
                "file": audio_file,
                "emotion": row["emotion"],
                "sentiment": row["sentiment"],
                "speaker": row["speaker"],
                "language_pair": language_pair,
                "language": target_language,
                "dialogue_id": row["dia_id"],
                "utterance_id": row["utt_id"],
                "season": row["season"],
                "episode": row["episode"],
            }

            # Add text transcription based on target language
            text_content = ""
            if has_target_text and target_text_col:
                text_val = row[target_text_col]
                if (
                    text_val is not None
                    and str(text_val).strip() != "nan"
                    and str(text_val).strip() != ""
                ):
                    text_content = str(text_val)

            # Always include text column (in target language)
            output_row["text"] = text_content

            # Also include English transcription as separate column if available and different from target
            if has_english_text:
                english_val = row[english_text_col]
                if (
                    english_val is not None
                    and str(english_val).strip() != "nan"
                    and str(english_val).strip() != ""
                ):
                    output_row["English"] = str(english_val)
                else:
                    output_row["English"] = ""

            output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)
    print(f"  Generated {len(output_df)} audio file entries for {target_language}")

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MELD-ST dataset for nkululeko"
    )
    parser.add_argument(
        "--data_dir", default="MELD-ST", help="Path to MELD-ST data directory"
    )
    parser.add_argument(
        "--output_dir",
        default=os.getcwd(),
        help="Output directory for processed CSV files",
    )
    parser.add_argument(
        "--language_pairs",
        nargs="+",
        choices=["ENG_DEU", "ENG_JPN"],
        default=["ENG_DEU", "ENG_JPN"],
        help="Language pairs to process",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each language pair
    for lang_pair in args.language_pairs:
        print(f"\n=== Processing {lang_pair} ===")

        lang_dir = data_dir / lang_pair
        if not lang_dir.exists():
            print(f"Warning: {lang_pair} directory not found, skipping...")
            continue

        # Define file mappings
        csv_files = {
            "train": f"{lang_pair.split('_')[1].lower()}_train.csv",
            "valid": f"{lang_pair.split('_')[1].lower()}_valid.csv",
            "test": f"{lang_pair.split('_')[1].lower()}_test.csv",
        }

        # Get the languages for this pair
        languages = ["ENG", lang_pair.split("_")[1]]  # ENG and DEU/JPN

        # Process each language separately
        for language in languages:
            lang_code = language.lower()
            all_splits_data = []

            # Process each split for this language
            for split_name, csv_filename in csv_files.items():
                csv_path = lang_dir / csv_filename

                if not csv_path.exists():
                    print(f"Warning: {csv_path} not found, skipping {split_name} split")
                    continue

                # Process this split for this language
                split_df = process_meld_st_split(
                    csv_path=csv_path,
                    audio_base_path=lang_dir,
                    language_pair=lang_pair,
                    split_name=split_name,
                    target_language=language,
                )

                # Add split information
                split_df["split"] = split_name
                all_splits_data.append(split_df)

                # Save individual split file for this language
                output_filename = (
                    f"meld_st_{lang_pair.lower()}_{lang_code}_{split_name}.csv"
                )
                output_path = output_dir / output_filename

                # Select columns for nkululeko compatibility (include transcription columns)
                base_columns = [
                    "file",
                    "emotion",
                    "sentiment",
                    "speaker",
                    "language",
                    "language_pair",
                ]

                # Always include text column since we now always add it
                transcription_cols = ["text"]

                # Add English column if it exists in dataframe
                if "English" in split_df.columns:
                    transcription_cols.append("English")

                nkululeko_columns = base_columns + transcription_cols
                available_columns = [
                    col for col in nkululeko_columns if col in split_df.columns
                ]

                split_df[available_columns].to_csv(output_path, index=False)
                print(f"  Saved: {output_path}")
                print(f"  Columns included: {available_columns}")

                # Verify text column content
                if "text" in split_df.columns:
                    non_empty_text = (
                        split_df["text"].apply(lambda x: str(x).strip() != "").sum()
                    )
                    print(
                        f"  Text column ({language}): {non_empty_text}/{len(split_df)} entries have non-empty text"
                    )

                if "English" in split_df.columns:
                    non_empty_english = (
                        split_df["English"].apply(lambda x: str(x).strip() != "").sum()
                    )
                    print(
                        f"  English column: {non_empty_english}/{len(split_df)} entries have non-empty text"
                    )

            # Combine all splits for this language and save
            if all_splits_data:
                combined_df = pd.concat(all_splits_data, ignore_index=True)
                combined_output_path = (
                    output_dir / f"meld_st_{lang_pair.lower()}_{lang_code}_all.csv"
                )

                # Save with all available columns including transcriptions
                base_columns = [
                    "file",
                    "emotion",
                    "sentiment",
                    "speaker",
                    "language",
                    "language_pair",
                ]

                # Always include text column since we now always add it
                transcription_cols = ["text"]

                # Add English column if it exists in combined dataframe
                if "English" in combined_df.columns:
                    transcription_cols.append("English")

                all_columns = base_columns + transcription_cols
                available_columns = [
                    col for col in all_columns if col in combined_df.columns
                ]

                combined_df[available_columns].to_csv(combined_output_path, index=False)
                print(f"  Saved combined: {combined_output_path}")
                print(f"  Combined columns included: {available_columns}")

                # Verify text column content in combined file
                if "text" in combined_df.columns:
                    non_empty_text = (
                        combined_df["text"].apply(lambda x: str(x).strip() != "").sum()
                    )
                    print(
                        f"  Combined text column ({language}): {non_empty_text}/{len(combined_df)} entries have non-empty text"
                    )

                if "English" in combined_df.columns:
                    non_empty_english = (
                        combined_df["English"]
                        .apply(lambda x: str(x).strip() != "")
                        .sum()
                    )
                    print(
                        f"  Combined English column: {non_empty_english}/{len(combined_df)} entries have non-empty text"
                    )

                # Print statistics for this language
                print(f"\n  === {lang_pair} {language} Statistics ===")
                print(f"  Total samples: {len(combined_df)}")
                print(f"  Emotions: {sorted(combined_df['emotion'].unique())}")
                print(f"  Sentiments: {sorted(combined_df['sentiment'].unique())}")
                print(f"  Speakers: {len(combined_df['speaker'].unique())} unique")
                print("  Split distribution:")
                print(f"    {combined_df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
