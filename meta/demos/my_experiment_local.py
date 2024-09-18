# main.py
# Demonstration code to use the ML-experiment framework

import sys

sys.path.append("./src")
import configparser

import experiment as exp
from util import Util


def main(config_file):
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    util = Util()
    util.debug(f"running {expr.name}")

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()

    # extract features
    expr.extract_feats()

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    expr.run()

    print("DONE")


if __name__ == "__main__":
    #    main('./demos/exp_danish_local.ini')
    #    main('./demos/exp_emodb_wav2vec.ini')
    #    main('./demos/exp_cross_wav2vec1pager.ini')
    #    main('./demos/exp_emodb_local.ini')
    main("./demos/exp_cross_local.ini")
#    main('./demos/exp_bdtgfir pul_local.ini')
#    main(sys.argv[1]) # use this if you want to state the config file path on command line
