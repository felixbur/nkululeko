# main.py
# Demonstration code to use the ML-experiment framework

import sys
sys.path.append("./src")
import nkululeko.experiment as exp
import configparser
from nkululeko.util import Util
from nkululeko.constants import VERSION

def main(config_file):
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # create a new experiment
    expr = exp.Experiment(config)
    util = Util()
    util.debug(f'running {expr.name}, nkululeko version {VERSION}')

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')

    # extract features
    expr.extract_feats()

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    expr.run()

    print('DONE')

if __name__ == "__main__":
    main('./tests/exp_cross.ini')