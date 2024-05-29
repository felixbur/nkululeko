# resample.py
# change the sampling rate for audio file or INI file (train, test, all)

import argparse
import configparser
import os
import pandas as pd
import audformat
from nkululeko.augmenting.resampler import Resampler
from nkululeko.utils.util import Util

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.files import find_files


def main(src_dir):
    parser = argparse.ArgumentParser(
        description="Call the nkululeko RESAMPLE framework."
    )
    parser.add_argument("--config", default=None,
                        help="The base configuration")
    parser.add_argument("--file", default=None,
                        help="The input audio file to resample")
    parser.add_argument(
        "--folder",
        default=None,
        help="The input directory containing audio files and subdirectories to resample",
    )
    parser.add_argument(
        "--replace", action="store_true", help="Replace the original audio file"
    )

    args = parser.parse_args()

    if args.file is None and args.folder is None and args.config is None:
        print(
            "ERROR: Either --file, --folder, or --config argument must be provided."
        )
        exit()

    if args.file is not None:
        # Load the audio file into a DataFrame
        files = pd.Series([args.file])
        df_sample = pd.DataFrame(index=files)
        df_sample.index = audformat.utils.to_segmented_index(
            df_sample.index, allow_nat=False
        )

        # Resample the audio file
        util = Util("resampler", has_config=False)
        util.debug(f"Resampling audio file: {args.file}")
        rs = Resampler(df_sample, not_testing=True, replace=args.replace)
        rs.resample()
    elif args.folder is not None:
        # Load all audio files in the directory and its subdirectories into a DataFrame
        files = find_files(args.folder, relative=True, ext=["wav"])
        files = pd.Series(files)
        df_sample = pd.DataFrame(index=files)
        df_sample.index = audformat.utils.to_segmented_index(
            df_sample.index, allow_nat=False
        )

        # Resample the audio files
        util = Util("resampler", has_config=False)
        util.debug(f"Resampling audio files in directory: {args.folder}")
        rs = Resampler(df_sample, not_testing=True, replace=args.replace)
        rs.resample()
    else:
        # Existing code for handling INI file
        config_file = args.config

        # Test if the configuration file exists
        if not os.path.isfile(config_file):
            print(f"ERROR: no such file: {config_file}")
            exit()

        # Load one configuration per experiment
        config = configparser.ConfigParser()
        config.read(config_file)
        # Create a new experiment
        expr = Experiment(config)
        module = "resample"
        expr.set_module(module)
        util = Util(module)
        util.debug(
            f"running {expr.name} from config {config_file}, nkululeko version"
            f" {VERSION}"
        )

        if util.config_val("EXP", "no_warnings", False):
            import warnings

            warnings.filterwarnings("ignore")

        # Load the data
        expr.load_datasets()

        # Split into train and test
        expr.fill_train_and_tests()
        util.debug(
            f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}"
        )

        sample_selection = util.config_val(
            "RESAMPLE", "sample_selection", "all")
        if sample_selection == "all":
            df = pd.concat([expr.df_train, expr.df_test])
        elif sample_selection == "train":
            df = expr.df_train
        elif sample_selection == "test":
            df = expr.df_test
        else:
            util.error(
                f"unknown selection specifier {sample_selection}, should be [all |"
                " train | test]"
            )
        util.debug(f"resampling {sample_selection}: {df.shape[0]} samples")
        replace = util.config_val("RESAMPLE", "replace", "False")
        rs = Resampler(df, replace=replace)
        rs.resample()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)
