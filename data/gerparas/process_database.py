#!/usr/bin/env python3
# process_database.py --> GerPaRas

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="input dir", default="./GerPaRaS/")
    parser.add_argument("--out_dir", help="output output dir", default="./")
    args = parser.parse_args()

    # Read valence, arousal, and dominance from the database
    data_dir = args.data_dir

    val_df = pd.read_csv(data_dir + "df_valence.csv").rename(columns={
        "Media File": "file", "orig": "valence", 'scrambled': 'valence_scrambled'})
    aro_df = pd.read_csv(data_dir + "df_arousal.csv").rename(columns={
        "Media File": "file", "orig": "arousal", 'scrambled': 'arousal_scrambled'})
    dom_df = pd.read_csv(data_dir + "df_dominance.csv").rename(columns={
        "Media File": "file", "orig": "dominance", 'scrambled': 'dominance_scrambled'})

    # standardize original score and scrambled score from [-6, 6] to [0, 1]
    val_df["valence"] = val_df["valence"].apply(lambda x: (x + 6) / 12)
    aro_df["arousal"] = aro_df["arousal"].apply(lambda x: (x + 6) / 12)
    dom_df["dominance"] = dom_df["dominance"].apply(lambda x: (x + 6) / 12)

    val_df["valence_scrambled"] = val_df["valence_scrambled"].apply(lambda x: (x + 6) / 12) 
    aro_df["arousal_scrambled"] = aro_df["arousal_scrambled"].apply(lambda x: (x + 6) / 12)
    dom_df["dominance_scrambled"] = dom_df["dominance_scrambled"].apply(lambda x: (x + 6) / 12)

    # Merge the three dataframes, use `orig` as value for `valence`, `arousal`, and `dominance`
    # so that a file has values of val, aro, and dom for original only
    df = val_df.merge(aro_df, on="file").merge(dom_df, on="file")
    # Drop the `scrambled` columns
    # orig_df = df.drop(columns=["val_scrambled", "aro_scrambled", "dom_scrambled"])

    # get speaker id, this is the first word of the file name before "_"
    df["speaker"] = df["file"].apply(lambda x: x.split("_")[0])

    # rename speaker goering to goering_eckhardt
    df["speaker"] = df["speaker"].apply(lambda x: "goering_eckhardt" if x == "goering" else x)
        
    # add speaker id as parent folder to file name
    df["file"] = df["speaker"] + "/" + df["file"]

    # set speakers gauland and weidel as test set
    test_speakers = ["gauland", "weidel"]

    # set all other speakers as train set
    train_speakers = [s for s in df["speaker"].unique() if s not in test_speakers]

    # set train and test set
    train_df = df[df["speaker"].isin(train_speakers)]
    test_df = df[df["speaker"].isin(test_speakers)]

    # save train and test set to csv
    train_df.to_csv(args.out_dir + "gerparas_train.csv", index=False)
    test_df.to_csv(args.out_dir + "gerparas_test.csv", index=False)
    df.to_csv(args.out_dir + "gerparas.csv", index=False)

    # print length of train and test set and total
    print(f"Total length: {len(df)}, Train set: {len(train_df)}, Test set: {len(test_df)}")


if __name__ == "__main__":
    main()
