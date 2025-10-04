# process_database.py for English Dialect Classification 

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# ----------- STEP 1: Process line_index_all.csv ------------

def process_line_index_all(input_csv, output_csv):
    print(f"Processing {input_csv} ...")
    df = pd.read_csv(input_csv, header=None, names=['line_ID', 'audio_with_out_wav', 'transcription'])
    df['speaker'] = df['audio_with_out_wav'].str.split('_').str[:2].str.join('_')
    df['dialect'] = df['audio_with_out_wav'].str.split('_').str[0].str[:-1]
    df['gender'] = df['audio_with_out_wav'].str.split('_').str[0].str[-1]
    print("Columns:", df.columns)
    print(df.head())
    df.to_csv(output_csv, index=False)
    print(f"Saved processed CSV to {output_csv}")

# ----------- STEP 2: Add path and enrich data ------------

def process_add_path(data_dir, input_csv, output_csv):
    print(f"Enriching data with audio file paths in {data_dir} ...")
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir {data_dir} does not exist!")
    trans_df = pd.read_csv(input_csv)
    trans_df = trans_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    dialect_map = {
        "northern_english_female": "northern",
        "northern_english_male": "northern",
        "scottish_english_female":"scottish",
        "scottish_english_male":"scottish",
        "welsh_english_female":"welsh",
        "welsh_english_male":"welsh",
        "midlands_english_female":"midlands",
        "midlands_english_male":"midlands",
        "irish_english_female":"irish",
        "irish_english_male":"irish",
        "southern_english_female":"southern",
        "southern_english_male":"southern",
    }
    gender_map = {"f": "female", "m": "male"}
    paths = list(data_dir.glob('**/*.wav'))
    if len(paths) == 0:
        raise FileNotFoundError("No audio files found in data_dir.")
    text_lookup = pd.Series(trans_df['transcription'].values, index=trans_df['audio_with_out_wav']).to_dict()
    file = [p for p in paths]
    dialect = [dialect_map.get(p.parent.name, "unknown") for p in paths]
    speaker = ['_'.join(p.stem.split('_')[:2]) for p in paths]
    gender = [gender_map.get(p.stem.split('_')[0][-1], "unknown") for p in paths]
    text = []
    for p in paths:
        filename = p.stem
        found_text = text_lookup.get(filename, "TEXT_NOT_FOUND")
        text.append(found_text)
    df = pd.DataFrame(data={'file': file, 'dialect': dialect, 'speaker': speaker, 'gender': gender, 'text': text})
    not_found_count = text.count("TEXT_NOT_FOUND")
    if not_found_count > 0:
        print(f"Warning: {not_found_count} files did not have a corresponding text entry found in the CSV.")
        not_found_files = [paths[i].stem for i, t in enumerate(text) if t == "TEXT_NOT_FOUND"]
        print(f"Example filename(s) without text: {not_found_files[:5]}")
    else:
        print("All audio files matched with text.")
    df.to_csv(output_csv, index=False)
    print(f"Saved enriched CSV to {output_csv}")

# ----------- STEP 3: Split dataset by dialect ------------

def split_dataset(input_csv, db_name='en-dialect'):
    print("Splitting dataset by dialect and speaker ...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV file {input_csv} does not exist")
    
    result_df = pd.read_csv(input_csv)
    
    # Detect which column is used for sorting (compatible with Step 1 and Step 2 outputs)
    if 'audio_with_out_wav' in result_df.columns:
        sort_column = 'audio_with_out_wav'
    elif 'file' in result_df.columns:
        sort_column = 'file'
    else:
        raise ValueError("CSV must contain either 'audio_with_out_wav' or 'file' column")
    
    result_df = result_df.sort_values(sort_column).reset_index(drop=True)
    
    train_frac = 0.8
    dev_frac = 0.1
    test_frac = 0.1
    
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # Step 1: Group by Dialect
    dialects = sorted(result_df['dialect'].unique())
    print(f"\nFound {len(dialects)} dialects: {dialects}")
    
    for dialect in dialects:
        print(f"\nProcessing dialect: {dialect}")
        dialect_df = result_df[result_df['dialect'] == dialect]
        
        # Step 2: Within each dialect, get the list of speakers and randomly split
        speakers = sorted(dialect_df['speaker'].unique())
        n_speakers = len(speakers)
        print(f"  Total speakers in {dialect}: {n_speakers}")
        
        # For reproducibility, set random seed and shuffle speaker list
        import random
        speakers_shuffled = speakers.copy()
        random.seed(42)  
        random.shuffle(speakers_shuffled)
        
        # Calculate how many speakers each set should contain.
        n_train_speakers = int(n_speakers * train_frac)
        n_dev_speakers = int(n_speakers * dev_frac)
        # split test
        n_test_speakers = n_speakers - n_train_speakers - n_dev_speakers
        
        # Assign speakers to each set
        train_speakers = speakers_shuffled[:n_train_speakers]
        dev_speakers = speakers_shuffled[n_train_speakers:n_train_speakers + n_dev_speakers]
        test_speakers = speakers_shuffled[n_train_speakers + n_dev_speakers:]
        
        print(f"  Train speakers: {len(train_speakers)}")
        print(f"  Dev speakers: {len(dev_speakers)}")
        print(f"  Test speakers: {len(test_speakers)}")
        
        # split based on speakers
        dialect_train = dialect_df[dialect_df['speaker'].isin(train_speakers)]
        dialect_dev = dialect_df[dialect_df['speaker'].isin(dev_speakers)]
        dialect_test = dialect_df[dialect_df['speaker'].isin(test_speakers)]
        
        print(f"  Train samples: {len(dialect_train)}")
        print(f"  Dev samples: {len(dialect_dev)}")
        print(f"  Test samples: {len(dialect_test)}")
        
        # merge to overall df
        train_df = pd.concat([train_df, dialect_train])
        dev_df = pd.concat([dev_df, dialect_dev])
        test_df = pd.concat([test_df, dialect_test])

    # Sort and reset index
    train_df = train_df.sort_values(sort_column).reset_index(drop=True)
    dev_df = dev_df.sort_values(sort_column).reset_index(drop=True)
    test_df = test_df.sort_values(sort_column).reset_index(drop=True)
    
    # save to csv
    train_file = f'{db_name}_train.csv'
    dev_file = f'{db_name}_dev.csv'
    test_file = f'{db_name}_test.csv'
    
    train_df.to_csv(train_file, index=False)
    dev_df.to_csv(dev_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    # print summary
    total_len = len(result_df)
    train_len = len(train_df)
    dev_len = len(dev_df)
    test_len = len(test_df)
    
    print("\n" + "="*50)
    print("Split complete!")
    print(f"Total sample size: {total_len}")
    print(f"Training set: {train_len} ({train_len/total_len:.1%})")
    print(f"Dev set: {dev_len} ({dev_len/total_len:.1%})")
    print(f"Test set: {test_len} ({test_len/total_len:.1%})")
    
    # print dialect distribution in each set
    print("\nDialect distribution:")
    for split_name, split_df in [('Train', train_df), ('Dev', dev_df), ('Test', test_df)]:
        print(f"\n{split_name} set:")
        dialect_counts = split_df['dialect'].value_counts().sort_index()
        for dialect, count in dialect_counts.items():
            print(f"  {dialect}: {count} samples")
    
    print("="*50)

# ----------- MAIN CLI ------------

def main():
    parser = argparse.ArgumentParser(description="EN-DIALECT Database Processing Pipeline")
    parser.add_argument('--data_dir', type=str, default='./02_process', help='Path to the data directory containing .wav files')
    parser.add_argument('--raw_csv', type=str, default='./line_index_all.csv', help='Raw CSV file to process')
    parser.add_argument('--skip_step1', action='store_true', help='Skip Step 1: initial CSV processing')
    parser.add_argument('--skip_step2', action='store_true', help='Skip Step 2: adding file paths')
    parser.add_argument('--skip_step3', action='store_true', help='Skip Step 3: splitting dataset')
    args = parser.parse_args()

    step1_csv = './line_index_all_process_1.csv'
    step2_csv = 'en-dialect_all_add_path.csv'

    if not args.skip_step1:
        process_line_index_all(args.raw_csv, step1_csv)
    else:
        print("Skipping Step 1")

    if not args.skip_step2:
        process_add_path(args.data_dir, step1_csv, step2_csv)
    else:
        print("Skipping Step 2")

    if not args.skip_step3:
        split_dataset(step2_csv)
    else:
        print("Skipping Step 3")

if __name__ == '__main__':
    main()
