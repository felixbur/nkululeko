#!/usr/bin/env python3
"""
Process the JESS database and generate jess.csv
with columns: file, speaker, gender, age_group, text

The script finds all .wav files in the JESS directory and extracts metadata from the file paths.

Requirements:
    pip install pandas
"""

import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt


def extract_metadata_from_path(file_path, base_path):
    """
    Extract metadata from the file path structure.

    Path structure examples:
    - osfstorage-archive/sentences/sentences_young/Sdiepostfrau_fKS25.wav
    - osfstorage-archive/vowels/aeiou_old/o_fRP71.wav
    - osfstorage-archive/read text/Nordwind_young/Nordwind_fKS25.wav
    - osfstorage-archive/semi-spontaneous speech/kitchen_farm_old/ki_fSS70.wav

    Filename pattern: <content>_<gender><speaker_id><age>.wav
    where gender is 'f' (female) or 'm' (male)
    """
    # Get relative path from base
    rel_path = file_path.relative_to(base_path)

    # Extract age_group from path
    path_str = str(rel_path)
    if '_young' in path_str or '/young/' in path_str:
        age_group = 'young'
    elif '_old' in path_str or '/old/' in path_str:
        age_group = 'old'
    else:
        age_group = 'unknown'

    # Extract text type from path
    if 'sentences' in path_str:
        text_type = 'sentence'
    elif 'vowels' in path_str or 'aeiou' in path_str:
        text_type = 'vowel'
    elif 'syllables' in path_str or 'aba_igi' in path_str:
        text_type = 'syllable'
    elif 'read text' in path_str or 'Nordwind' in path_str:
        text_type = 'read_text'
    elif 'semi-spontaneous' in path_str or 'spontaneous' in path_str:
        text_type = 'spontaneous'
    else:
        text_type = 'unknown'

    # Extract gender and speaker_id from filename
    filename = file_path.stem  # filename without extension

    """
    All file names denote speech content and speaker
    information including sex (f/m), ID and age. For example, sound file “aba_fAB75.wav” contains the
    syllable “aba” uttered by the female speaker “AB” who was 75 years old at the time of recording.
    """
    # Extract age from filename 
    age_match = re.search(r'(\d{2})$', filename)
    age = int(age_match.group(1)) if age_match else -1

    # Pattern: content_<gender><speaker_id>
    # Examples: Sdiepostfrau_fKS25, o_fRP71, Nordwind_fKS25
    match = re.search(r'_([fm])([A-Z]{2}\d+)', filename)

    if match:
        gender = 'female' if match.group(1) == 'f' else 'male'
        speaker_id = match.group(2)
    else:
        gender = 'unknown'
        speaker_id = 'unknown'

    return {
        'file': str(rel_path),
        'speaker': speaker_id,
        'gender': gender,
        'age_group': age_group,
        'age': age,
        'text': text_type
    }


def process_jess_database():
    """
    Scan all .wav files in the JESS directory and create a CSV with metadata.

    Output columns:
    - file: relative path to audio file from jess directory
    - speaker: speaker ID (extracted from filename)
    - gender: female/male
    - age_group: old/young
    - text: type of speech (read_text, spontaneous, sentence, syllable, vowel)
    """
    # Base path is the jess directory
    base_path = Path(__file__).parent
    audio_dir = base_path / "osfstorage-archive"
    output_path = base_path / "jess.csv"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found at: {audio_dir}")

    print(f"Scanning for .wav files in: {audio_dir}")

    # Find all .wav files
    wav_files = list(audio_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} .wav files")

    if len(wav_files) == 0:
        raise ValueError("No .wav files found in the directory")

    # Extract metadata from each file
    data = []
    for wav_file in wav_files:
        metadata = extract_metadata_from_path(wav_file, base_path)
        data.append(metadata)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by file path for consistent ordering
    df = df.sort_values('file').reset_index(drop=True)

    # Display statistics
    print(f"\nDataset statistics:")
    print(f"Total files: {len(df)}")
    print(f"\nGender distribution:")
    print(df['gender'].value_counts())
    print(f"\nAge group distribution:")
    print(df['age_group'].value_counts())
    print(f"\nText type distribution:")
    print(df['text'].value_counts())
    print(f"\nNumber of unique speakers: {df['speaker'].nunique()}")
    # plot age distribution
    print(f"\nAge distribution:")
    print(df['age'].describe())
    ax = df['age'].hist(bins=range(0, 101, 5))
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of speakers') 
    ax.set_title('Age Distribution of Speakers')
    # save to file
    plt.savefig("age_distribution.png")
    plt.close()

    # Show sample rows
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string())

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Output saved to: {output_path}")

    return df


if __name__ == "__main__":
    try:
        df = process_jess_database()
        print("\n✓ Successfully processed the database!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
