""" process_database.py for EAED dataset

file name format: <speaker>_<emotion>_<(number)>.wav
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):
    # check if data_dir exists
    if not Path(data_dir).is_dir():
        print(f"ERROR: no such directory {data_dir}")
        return
    # create output dir if not exist
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir()

    # read all wav files
    wavs = find_files(data_dir, relative=True, ext=["wav"])
    print(f"Found {len(wavs)} wav files.")

    # building dataframe from wavs list
    df = pd.DataFrame(wavs, columns=["file"])
    df["file"] = df["file"].apply(lambda x: str(x))

    # get emotion from file basename, make all smallcase
    df["emotion"] = df["file"].apply(lambda x: x.split("_")[1].lower())

    # get speaker from file basename, firs string before _
    df["speaker"] = df["file"].apply(lambda x: Path(x).name.split("_")[0])

    # add language = arabic
    df["language"] = "arabic"

    # make speaker independent partition
    speakers = df["speaker"].unique()
    train_speakers, dev_speakers = train_test_split(speakers, test_size=0.2)
    dev_speakers, test_speakers = train_test_split(dev_speakers, test_size=0.5)

    # loop over train, dev, and test and save as csv
    for set_name in ["train", "dev", "test"]:
        df_set = df[df["speaker"].isin(eval(f"{set_name}_speakers"))]
        df_set.to_csv(Path(output_dir, f"eaed_{set_name}.csv"), index=False)
        print(
            f"Saved {len(df_set)} samples to {Path(output_dir, f'eaed_{set_name}.csv')}"
        )

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EAED dataset")
    parser.add_argument(
        "--data_dir", type=str, default="./EAED/", help="Path to the EAED dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="Path to the output directory"
    )
    args = parser.parse_args()
    process_database(args.data_dir, args.output_dir)
