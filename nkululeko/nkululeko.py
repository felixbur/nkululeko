# nkululeko.py
# Entry script to do a Nkululeko experiment

import nkululeko.experiment as exp
import configparser
from nkululeko.util import Util
import os.path
from nkululeko.constants import VERSION
import argparse

def main(src_dir):
    parser = argparse.ArgumentParser(description='Call the nkululeko framework.')
    parser.add_argument('--config', default='exp.ini', help='The base configuration')
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config    
    else:
        config_file = f'{src_dir}/exp.ini'

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f'ERROR: no such file: {config_file}')
        exit()

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
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd) # use this if you want to state the config file path on command line
