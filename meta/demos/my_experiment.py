# main.py
# Demonstration code to use the ML-experiment framework

import sys

sys.path.append("./nkululeko/src")
import configparser
import os.path

import constants
import experiment as exp
from util import Util


def main(config_file):
    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    util = Util()
    util.debug(f"running {expr.name}, nkululeko version {constants.VERSION}")

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
    expr.run()

    print("DONE")


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main("./nkululeko/demos/exp_emodb.ini")
#    main(sys.argv[1]) # use this if you want to state the config file path on command line
