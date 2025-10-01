# neurovoz, see README.md
# convert the data into nkululeko CSV format


import os
import pandas as pd
import numpy as np

# load the datatable for hc and pd from the files data/metadata/metadata_hc.csv and data/metadata/metadata_pd.csv respectively


def load_data():
    """
    Load the dataset from the CSV file.
    """
    df_hc = pd.read_csv("data/metadata/metadata_hc.csv")
    df_pd = pd.read_csv("data/metadata/metadata_pd.csv")
    return df_hc, df_pd


# re_arrange the columns to match nkululeko format
# in the original csv files, the file_path is in the column "Audio" at the very end
# in nkululeko format, the file_path should be in the first column named "file"
# so we need to move the column "Audio" to the first position and rename it
def re_arrange_columns(df):
    # move the column "Audio" to the first position
    cols = df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    # rename the column "Audio" to "file"
    df = df.rename(columns={"Audio": "file"})
    return df


def main():
    df_hc, df_pd = load_data()
    print("HC data shape:", df_hc.shape)
    print("PD data shape:", df_pd.shape)
    df_hc = re_arrange_columns(df_hc)
    df_pd = re_arrange_columns(df_pd)
    # concatenate the two dataframes
    df = pd.concat([df_hc, df_pd], ignore_index=True)
    print("Combined data shape:", df.shape)
    # check if all files exist, else remove the row
    df["file_exists"] = df["file"].apply(lambda x: os.path.exists(x))
    n_missing = df["file_exists"].value_counts().get(False, 0)
    print("Number of missing files:", n_missing)
    df = df[df["file_exists"]]
    df = df.drop(columns=["file_exists"])
    print("Data shape after removing missing files:", df.shape)
    # save the dataframe to a csv file
    df.to_csv("neurovoz.csv", index=False)


if __name__ == "__main__":
    main()
