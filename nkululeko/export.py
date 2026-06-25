# export.py
# export the loaded training and test sets to it's own folder

import argparse
import configparser
import os
import shutil
import sys

import audeer
import audiofile
import pandas as pd

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.util import Util


def _export_file(file, file_dir, file_name, target_root, orig_root):
    """Build the destination path for an exported file and return its
    target-root-relative path."""
    wav_folder = f"{target_root}/{os.path.basename(os.path.normpath(orig_root))}"
    audeer.mkdir(wav_folder)
    new_rel_path = file_dir[file_dir.index(orig_root) + 1 + len(orig_root) :]
    new_file_path = f"{wav_folder}/{new_rel_path}"
    audeer.mkdir(new_file_path)
    new_file_name = f"{new_file_path}/{file_name}"
    return new_file_name


def _process_split(df, split, segments_as_files, target_root, orig_root):
    """Export the audio of one split and return the dataframe re-indexed to the
    exported file paths."""
    files = []
    for file, start, end in df.index.to_list():
        file_dir = os.path.dirname(file)
        if segments_as_files:
            signal, sampling_rate = audiofile.read(
                file,
                offset=start.total_seconds(),
                duration=(end - start).total_seconds(),
                always_2d=True,
            )
            file_name = os.path.splitext(file)[0] + "_" + start.total_seconds() + ".wav"
            new_file_name = _export_file(
                file, file_dir, file_name, target_root, orig_root
            )
            audiofile.write(new_file_name, signal, sampling_rate)
        else:
            file_name = os.path.basename(file)
            new_file_name = _export_file(
                file, file_dir, file_name, target_root, orig_root
            )
            if not os.path.exists(new_file_name):
                shutil.copyfile(file, new_file_name)
        files.append(os.path.relpath(new_file_name, target_root))
    df = df.set_index(df.index.set_levels(files, level="file"))
    df["split"] = split
    return df


def main():
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    config_file = args.config if args.config is not None else "exp.ini"

    try:
        if not os.path.isfile(config_file):
            print(f"ERROR: no such file: {config_file}")
            sys.exit(1)

        config = configparser.ConfigParser()
        config.read(config_file)
        expr = Experiment(config)
        util = Util("export")
        util.debug(
            f"running {expr.name} from config {config_file}, nkululeko version {VERSION}"
        )

        if util.config_val("EXP", "no_warnings", False):
            import warnings

            warnings.filterwarnings("ignore")

        # load the data
        expr.load_datasets()

        # split into train and test
        expr.fill_train_and_tests()
        util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

        # export
        df_train = expr.df_train
        df_test = expr.df_test
        target_root = util.config_val("EXPORT", "root", "./exported_data/")
        orig_root = util.config_val("EXPORT", "orig_root", None)
        data_name = util.config_val("EXPORT", "data_name", "export")
        segments_as_files = eval(util.config_val("EXPORT", "segments_as_files", "False"))
        audeer.mkdir(target_root)
        splits = {"train": df_train, "test": df_test}
        df_all = pd.DataFrame()
        for split, df in splits.items():
            df = _process_split(df, split, segments_as_files, target_root, orig_root)
            df_all = pd.concat([df_all, df])
        # remove encoded labels
        target = util.config_val("DATA", "target", "emotion")
        if "class_label" in df_all.columns:
            df_all = df_all.drop(columns=[target])
            df_all = df_all.rename(columns={"class_label": target})

        df_all.to_csv(f"{target_root}/{data_name}.csv")
        util.debug(f"saved {data_name}.csv to {target_root}, {df.shape[0]} samples.")

        print("DONE")
    except NkululukoError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
