# process_database --> BAVED
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/BAVED/process.py

import argparse
from html import parser
from pathlib import Path

import pandas as pd

emotion_level = ["low", "normal", "high"]
word_map = [
    "اعجبني",
    "لم يعجبني",
    "هذا",
    "الفيلم",
    "رائع",
    "مقول",
    "سيئ",
]

gender_map = {
    "M": "male",
    "F": "female"
}


def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='BAVED')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # read data
    paths = list(data_dir.glob('?/*.wav'))
    emo_levels = [emotion_level[int(p.stem.split('-')[4])] for p in paths]
    speakers = [p.stem.split('-')[0] for p in paths]
    genders = [gender_map[p.stem.split('-')[1].upper()] for p in paths]
    ages = [int(p.stem.split('-')[2]) for p in paths]
    words = [word_map[int(p.stem.split('-')[3])] for p in paths]
    languages = ["arabic" for p in paths]

    # convert to dataframe
    df = pd.DataFrame({"file": paths, "emotion_level": emo_levels,
                      "speakers": speakers, "gender": genders, "age": ages,
                       "word": words, "language": languages})

    # allocate speaker 50, 46, 102 , 55, 51
    test_speakers = ['46', '4', '54', '47', '51']
    df_test = df[df['speakers'].isin(test_speakers)]
    df_train = df.drop(df_test.index)

    # save to CSV
    df.to_csv(output_dir / 'baved.csv', index=False)
    df_train.to_csv(output_dir / 'baved_train.csv', index=False)
    df_test.to_csv(output_dir / 'baved_test.csv', index=False)

    print(f"BAVED: {len(df)} samples, {len(df_train)} train, {len(df_test)} test")

if __name__ == '__main__':
    main()
