#!/usr/bin/env python3
# process_database.py --> CLAC
# need to install openpyxl: pip install openpyxl

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="input dir", default="CLAC-Dataset")
    parser.add_argument("--out_dir", help="output dir", default=".")
    args = parser.parse_args()

    # Read metadata from excel file
    metadata = pd.read_excel(args.data_dir + "/metadata.xlsx", index_col=0)

    # rename columns
    metadata.rename(
        columns={
            "speakerID": "speaker",
            "worker_country": "country",
            "worker_region": "region",
            "age (years)": "age",
        },
        inplace=True,
    )

    # remove "education (years)" column
    metadata.drop(columns=["education (years)"], inplace=True)

    # add file with WAV extension
    metadata["file"] = metadata["speaker"] + ".wav"

    # print(metadata.head())
    print(metadata.head())

    # print length of metadata
    print(f"Length of metadata: {len(metadata)}, saved as metadata.csv")

    # save to csv file
    metadata.to_csv(args.out_dir + "/clac.csv")


if __name__ == "__main__":
    main()
