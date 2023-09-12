# process_database.py --> ekorpus
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/EESC/process.py
# need to install textgrid: pip install textgrid

import argparse
from pathlib import Path

import pandas as pd
import textgrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='EKORPUS')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob('*.wav'))

    labels = {}
    sentences = {}
    for path in tqdm(
        data_dir.glob("*.TextGrid"), desc="Processing annotations", total=len(paths)
    ):
        grid = textgrid.TextGrid.fromFile(path)
        labels[path.stem] = grid.getFirst("emotion")[0].mark
        sentences[path.stem] = grid.getFirst("sentence")[0].mark

    emotion = labels.values()
    language = ['estonian' for _ in range(len(paths))]

    df = pd.DataFrame(
        {"file": paths, "emotion": emotion, "language": language})

    # split for training and test
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['emotion'])


    # save to CSV
    df.to_csv(output_dir / 'ekorpus.csv', index=False)
    df_train.to_csv(output_dir / 'ekorpus_train.csv', index=False)
    df_test.to_csv(output_dir / 'ekorpus_test.csv', index=False)

    print(f"EKORPUS: {len(df)} samples, {len(df_train)} train, {len(df_test)} test")


if __name__ == '__main__':
    main()
