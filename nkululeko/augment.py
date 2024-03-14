# augment.py
# augment the training sets

import argparse
import pandas as pd
import os
import ast
from nkululeko.experiment import Experiment
import configparser
from nkululeko.utils.util import Util
from nkululeko.constants import VERSION


def doit(config_file):
    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    # create a new experiment
    expr = Experiment(config)
    module = "augment"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    filename = util.config_val("AUGMENT", "result", "augmented.csv")
    filename = f"{expr.data_dir}/{filename}"

    if os.path.exists(filename):
        util.debug("files already augmented")
    else:
        # load the data
        expr.load_datasets()

        # split into train and test
        expr.fill_train_and_tests()
        util.debug(
            f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}"
        )

        # augment
        augmentings = util.config_val("AUGMENT", "augment", False)
        got_one = False
        if augmentings:
            augmentings = ast.literal_eval(augmentings)
            results = []
            if "traditional" in augmentings:
                df1 = expr.augment()
                results.append(df1)
                got_one = True
            if "random_splice" in augmentings:
                df2 = expr.random_splice()
                results.append(df2)
                got_one = True
        if not augmentings:
            util.error("no augmentation selected")
        if not got_one:
            util.error(f"invalid augmentation(s): {augmentings}")
        df_ret = pd.DataFrame()
        df_ret = pd.concat(results)
        # remove encoded labels
        target = util.config_val("DATA", "target", "emotion")
        if "class_label" in df_ret.columns:
            df_ret = df_ret.drop(columns=[target])
            df_ret = df_ret.rename(columns={"class_label": target})
        # save file
        df_ret.to_csv(filename)
        util.debug(f"saved augmentation table to {filename} to {expr.data_dir}")
        print("DONE")


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"
    doit(config_file)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
