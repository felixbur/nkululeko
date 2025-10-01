import argparse
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser(description='Process EN-DIALECT database')
parser.add_argument('--data_dir', type=str, default='./02_process',
                    help='path to the data directory')
args = parser.parse_args()

data_dir = Path(args.data_dir)
assert data_dir.exists()

#add transcription from csv file save from step1
trans_df = pd.read_csv('line_index_all_process_1.csv')

# clean all spaces in texts
trans_df = trans_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#dialect map for short name
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

gender_map = {
    "f": "female",
    "m": "male"
}



def main():
    input_dir = Path(args.data_dir)
    paths = list(input_dir.glob('**/*.wav'))
    if len(paths) == 0:
        raise FileNotFoundError("No audio files found.")

    try:
        #find 
        text_lookup = pd.Series(trans_df['transcription'].values, index=trans_df['audio_with_out_wav']).to_dict()
        print(f"load {len(text_lookup)} rows of text")
        #check if the file name is right
        print(f"samples in csv: {list(text_lookup.keys())[:3]}")
    except KeyError as e:
        print(f"error：no column in csv file {e}")
        print(f"real column name in CSV: {list(trans_df.columns)}")
        return

    
    # Use list comprehensions to efficiently process data
    file = [p for p in paths]
    dialect = [dialect_map[p.parent.name] for p in paths]
    speaker = ['_'.join(p.stem.split('_')[:2]) for p in paths]
    gender = [gender_map[p.stem.split('_')[0][-1]] for p in paths]

    # 2. Debugging Information: View audio file name format
    if len(paths) > 0:
        print(f"found {len(paths)} files")
        print(f"examples to check: {[p.stem for p in paths[:3]]}")
    
    # 3. Use the dictionary created above to retrieve the text for each file.
    text = []
    for p in paths:
        filename = p.stem  # filename without ".wav"
        found_text = text_lookup.get(filename, "TEXT_NOT_FOUND")
        text.append(found_text)

    # 4. build dataframe
    df = pd.DataFrame(data={'file': file, 'dialect': dialect, 'speaker': speaker, 'gender': gender, 'text': text})

    # check if there is any text not found
    not_found_count = text.count("TEXT_NOT_FOUND")
    if not_found_count > 0:
        print(f"warn： {not_found_count} ile did not have a corresponding text entry found in the CSV.")
        # for debug
        not_found_files = [paths[i].stem for i, t in enumerate(text) if t == "TEXT_NOT_FOUND"]
        print(f"Example of a filename without text: {not_found_files[:5]}")
    else:
        print("All audio files have successfully been matched with their corresponding text！")

    # --- SAVE THE ENTIRE DATAFRAME TO ONE FILE ---
    output_filename = 'en-dialect_all_add_path.csv'
    df.to_csv(output_filename, index=False)
    print(f"All data has been successfully processed and saved to '{output_filename}'")


if __name__ == "__main__":
    main()