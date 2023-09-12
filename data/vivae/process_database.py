# process_database.py --> VIVAE
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/VIVAE/process.py

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='VIVAE')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob('full_set/*.wav'))
    emotion = [p.stem.split('_')[1] for p in paths]
    speakers = [p.stem.split('_')[0] for p in paths]
    intensities = [p.stem.split('_')[2] for p in paths]

    df = pd.DataFrame({"file": paths, "emotion": emotion, "speaker": speakers, "intensity": intensities})

    # allocate speaker S05 and S06 to test set
    test_speakers = ['S05', 'S06']
    df_test = df[df['speaker'].isin(test_speakers)]
    df_train = df.drop(df_test.index)

    # save to CSV
    df.to_csv(output_dir / 'vivae.csv', index=False)
    df_train.to_csv(output_dir / 'vivae_train.csv', index=False)
    df_test.to_csv(output_dir / 'vivae_test.csv', index=False)


if __name__ == '__main__':
    main()
