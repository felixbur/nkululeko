# aug_train.py
# train with augmentations
import argparse
import ast
import configparser
import os.path

import numpy as np

from nkululeko.augment import doit as augment
from nkululeko.constants import VERSION
import nkululeko.experiment as exp
from nkululeko.utils.util import Util


def doit(config_file):
    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    module = "aug_train"
    expr.set_module(module)
    util = Util(module, context=expr.context)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    augmentings = util.config_val("AUGMENT", "augment", False)
    if not augmentings:
        util.error("no augmentation method specified")
    augmentings = ast.literal_eval(augmentings)
    augmentings = "_".join(augmentings)
    result_file = f"augmented_{augmentings}.csv"

    config = expr.context.config
    config["DATA"]["no_reuse"] = "True"
    config["FEATS"]["no_reuse"] = "True"
    config["AUGMENT"]["sample_selection"] = "train"
    config["AUGMENT"]["result"] = f"./{result_file}"
    tmp_config = "tmp.ini"
    with open(tmp_config, "w") as config_file:
        config.write(config_file)
    augment(tmp_config)
    databases = ast.literal_eval(config["DATA"]["databases"])
    aug_name = f"aug_{augmentings}"
    databases.append(aug_name)
    config["DATA"]["databases"] = str(databases)
    config["DATA"][aug_name] = f"{util.get_exp_dir()}/{result_file}"
    config["DATA"][f"{aug_name}.type"] = "csv"
    config["DATA"][f"{aug_name}.rename_speakers"] = "True"
    config["DATA"][f"{aug_name}.split_strategy"] = "train"
    util.set_config(config)
    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    # extract features
    expr.extract_feats()

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    reports, last_epochs = expr.run()
    result = expr.get_best_report(reports).result.test
    expr.store_report()
    print("DONE")
    return result, int(np.asarray(last_epochs).min())


def main():
    """Entrypoint for the nkululeko framework.

    This function parses command line arguments to determine the configuration file to use,
    and then calls the `doit` function with the specified configuration file.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    doit(args.config)


if __name__ == "__main__":
    main()  # use this if you want to state the config file path on command line
