# main.py
# Demonstration code to use the ML-experiment framework

import sys
sys.path.append("./src")
import experiment as exp
import configparser
from util import Util
import constants

def main(config_file):
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # create a new experiment
    expr = exp.Experiment(config)
    util = Util()
    util.debug(f'running {expr.name}, nkululeko version {constants.VERSION}')

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

    print('DONE')

if __name__ == "__main__":
    main('./tests/exp_bdtg_cross.ini')