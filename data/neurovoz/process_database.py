# neurovoz, see README.md
# convert the data into nkululeko CSV format


import os
import pandas as pd
import numpy as np
import audeer

# load the datatable for hc and pd from the files data/metadata/metadata_hc.csv and data/metadata/metadata_pd.csv respectively


def load_data():
    """
    Load the dataset from the CSV file.
    """
    current_dir = audeer.script_dir()
    df_hc = pd.read_csv(os.path.join(current_dir, "data/metadata/metadata_hc.csv"))
    df_pd = pd.read_csv(os.path.join(current_dir, "data/metadata/metadata_pd.csv"))
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
    # try to get all transcriptions from the files in the transcript folder
    transcript_folder = "data/transcriptions/"
    transcriptions = []
    tasks = []
    for index, row in df.iterrows():
        file_path = row['file']
        file_path = os.path.basename(file_path)  # get only the file name
        # get task
        task = file_path.split("_")[1]
        if task == "PATAKA":
            task = "pataka"
        elif len(task)== 2:
            task = "sustained"
        else:
            task = "text"
        tasks.append(task)
        # get transcription
        file_path = os.path.join(transcript_folder, audeer.replace_file_extension(file_path, "txt"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                transcriptions.append(f.read())
        else:
            transcriptions.append(None)
    df["task"] = tasks
    df["transcription"] = transcriptions
    print("Data shape after adding task and transcriptions:", df.shape)
    #df["ID"] = df["ID"].apply(lambda x: f"s_{str(x)}")
    # save the dataframe to a csv file

    """
    Now try to get GRBAS scores for all files
    They are hidden in the files under the grbas folder.
    Each recording tasks has its own file.
    In each file we got csv tables with the unique recording file name in the column
    "text_patient_id"
    and the GRBAS scores in the columns "G", "R", "B", "A", "S"
    """

    # Load all GRBAS CSV files and combine them
    grbas_folder = "data/grbas/"
    grbas_files = [f for f in os.listdir(grbas_folder) if f.endswith(".csv")]

    # Collect all GRBAS data into a single dataframe
    grbas_dfs = []
    for grbas_file in grbas_files:
        grbas_path = os.path.join(grbas_folder, grbas_file)
        grbas_df = pd.read_csv(grbas_path)
        grbas_dfs.append(grbas_df)

    # Concatenate all GRBAS dataframes
    grbas_combined = pd.concat(grbas_dfs, ignore_index=True)

    # Extract just the filename from the file path in main df
    df["filename"] = df["file"].apply(lambda x: os.path.basename(x).split('_', 1)[1])

    # Merge GRBAS scores with main dataframe based on filename
    df = df.merge(
        grbas_combined,
        left_on="filename",
        right_on="text_patient_id",
        how="left"
    )

    # Drop the redundant columns
    df = df.drop(columns=["text_patient_id", "filename"])

    print("Data shape after adding GRBAS scores:", df.shape)
    print("Number of rows with GRBAS scores:", df[["G", "R", "B", "A", "S"]].notna().any(axis=1).sum())
    df.rename(columns={'TOTAL':'total', 'COMMENTS ':'comments', 
       'GLOTOTIC ATTACK':'glototic_attack', 'INTENSITY':'intensity', 'SPEED':'speed', 'RESONANCE':'resonance', 'ARTICLES':'articles',
       'TONE':'tone', 'QUALITY PHONATION':'quality_phonation', 'INTELLIGIBILITY':'intelligibility', 'PROSODY':'prosody'},inplace=True)
    print("Number of rows with intelligibility scores:", df[["intelligibility"]].notna().any(axis=1).sum())
    
    # normalize the values in the column "intelligibility"
    normal_map = {"nromal":"normal", "normalnormal":"normal", "nnormal ":"normal",\
    "nnormal":"normal", "2 mild deficiency":"mild deficiency",\
    "milddefieicence":"mild deficiency",\
    "moderate": "moderate deficiency", "moderatedeficiency":"moderate deficiency"}
    df["intelligibility"] = df["intelligibility"].replace(normal_map)
    df.to_csv("neurovoz.csv", index=False)


if __name__ == "__main__":
    main()
