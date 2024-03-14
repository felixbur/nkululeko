# resample.py
# change the sampling rate for train and test splits

from nkululeko.experiment import Experiment
import configparser
from nkululeko.utils.util import Util
from nkululeko.constants import VERSION
import argparse
import os
import pandas as pd
from nkululeko.augmenting.resampler import Resampler


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    # create a new experiment
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

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    sample_selection = util.config_val("RESAMPLE", "sample_selection", "all")
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
    rs = Resampler(df)
    rs.resample()
    print("DONE")


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
