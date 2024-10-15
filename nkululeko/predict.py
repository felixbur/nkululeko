# predict.py
# use some model and add automatically predicted labels to train and test splits
# then save as a new dataset

"""This script is used to call the nkululeko PREDICT framework. 

It loads a configuration file, creates a new experiment,
and performs automatic prediction on the train and test datasets. The predicted labels are added to the datasets and
saved as a new dataset.

Usage: \n
    python3 -m nkululeko.predict [--config CONFIG_FILE] \n

Arguments: \n
    --config (str): The path to the base configuration file (default: exp.ini)
"""

import argparse
import configparser
import os

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util


def main():
    parser = argparse.ArgumentParser(
        description="Call the nkululeko PREDICT framework."
    )
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    config_file = args.config if args.config is not None else "exp.ini"

    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    config = configparser.ConfigParser()
    config.read(config_file)
    expr = Experiment(config)
    module = "predict"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    # process the data
    df = expr.autopredict()
    target = util.config_val("DATA", "target", "emotion")
    if "class_label" in df.columns:
        df = df.drop(columns=[target])
        df = df.rename(columns={"class_label": target})
    name = util.get_data_name() + "_predicted"
    df.to_csv(f"{expr.data_dir}/{name}.csv")
    util.debug(f"saved {name}.csv to {expr.data_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
