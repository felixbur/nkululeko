# nkululeko.py
# Entry script to do a Nkululeko experiment
import argparse
import configparser
from pathlib import Path

import numpy as np

import nkululeko.experiment as exp
from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


def doit(config_file):
    # test if the configuration file exists
    if not Path(config_file).is_file():
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    module = "nkululeko"
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
    cwd = Path(__file__).parent.absolute()
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--version", action="version", version=f"Nkululeko {VERSION}")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = cwd / "exp.ini"
    doit(config_file)


if __name__ == "__main__":
    main()  # use this if you want to state the config file path on command line
