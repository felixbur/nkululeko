#!/usr/bin/env python3
# process_database.py --> ENTERFACE


import argparse
from pathlib import Path

import pandas as pd

emotion_map = {
    "an": "anger",
    "di": "disgust",
    "fe": "fear",
    "ha": "happiness",
    "sa": "sadness",
    "su": "surprise",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ENTERFACE', help='Path to the database file')
    parser.add_argument('--output_dir', type=str, default='.', help='Path to the output directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()


    paths = list(data_dir.glob('**/*.wav'))

    if len(paths) == 0:
        paths = list(data_dir.glob('**/*.avi'))
    
        # manual correction
        for f in paths:
            if "subject 11" in f.parts:
                f.rename(f.parent / f.name.replace("s12", "s11"))
            if f.stem == 's16_su_3avi':
                f.rename(f.parent / f.name.replace("s16_su_3avi", "s16_su_3.avi"))
            if f.stem.startswith('s_3_'):
                f.rename(f.parent / f.name.replace("s_3", "s3"))

        # reload paths
        paths = list(data_dir.glob('**/*.avi'))
        # pd.DataFrame({"file": paths}).to_csv(output_dir / 'enterface_avi.csv', index=True, index_label='id')

        
    emotions = [p.parts[-2] if p.stem.startswith('s6_') else p.parts[-3] for p in paths]
    speakers = [p.stem.split('_')[0][1:] for p in paths]
    languages = ["english" for p in paths]

    df = pd.DataFrame({'file': paths, 'emotion': emotions, 'speaker': speakers, 'language': languages})


    # allocate speakers > 33 for test
    df_test = df[df['speaker'] > '40']
    df_train = df.drop(df_test.index)


    # save to csv
    df_train.to_csv(output_dir / 'enterface_train_avi.csv', index=False)
    df_test.to_csv(output_dir / 'enterface_test_avi.csv', index=False)
    
    # remove extension, change to wav
    df_train['file'] = df_train['file'].apply(lambda x: x.with_suffix('.wav'))
    df_test.loc[:,'file'] = df_test['file'].apply(lambda x: x.with_suffix('.wav'))

    df_train.to_csv(output_dir / 'enterface_train.csv', index=False)
    df_test.to_csv(output_dir / 'enterface_test.csv', index=False)


    print(f"Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")


if __name__ == '__main__':
    main()
