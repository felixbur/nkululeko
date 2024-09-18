# process_database.py for TurEV-DB

import argparse
from pathlib import Path

import pandas as pd

from nkululeko.utils.files import find_files


def process_database(data_dir, output_dir):  
    # check if data_dir exist
    if not Path(data_dir).exists():
        raise ValueError(f"data_dir {data_dir} does not exist")
    
    # check if output_dir exist, create if not
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    sound_source_dir = Path(data_dir) / "Sound Source"
    wavs = find_files(sound_source_dir, ext=["wav"], relative=True)
    print(f"Found {len(wavs)} files in {data_dir}")

    # read basename of each wav file, convert to dataframe
    basenames = [Path(wav) for wav in wavs]
    df = pd.DataFrame(basenames, columns=["file"])

    # read emotion from parent dir of basename
    df["emotion"] = [Path(wav).parent.stem.lower() for wav in wavs]

    # read speaker from basename first string before _
    df["speaker"] = [Path(wav).stem.split("_")[0] for wav in wavs]

    # split into train, val, test
    # there are six speakers, use one speaker "6783","1358" as val, test
    df_test = df[df["speaker"] == "6783"]
    df_dev = df[df["speaker"] == "1358"]

    # use the rest as train
    df_train = df[df["speaker"]!= "6783"]
    df_train = df_train[df_train["speaker"]!= "1358"]

    # save all splits to csv
    for split in ["train", "dev", "test"]:
        df_split = eval(f"df_{split}")
        df_split.to_csv(Path(output_dir) / f"turev_{split}.csv", index=False)
        print(f"Saved {split} to {output_dir}/turev_{split}.csv with shape {df_split.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./TurEV-DB", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default=".", help="Path to output directory")
    args = parser.parse_args()
    process_database(args.data_dir, args.output_dir)