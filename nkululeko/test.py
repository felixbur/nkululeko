# test.py
# Just use a database as test

from nkululeko.experiment import Experiment
import configparser
from nkululeko.util import Util
from  nkululeko.constants import VERSION
import argparse
import os

def main(src_dir):
    parser = argparse.ArgumentParser(description='Call the nkululeko framework.')
    parser.add_argument('--config', default='exp.ini', help='The base configuration')
    parser.add_argument('--outfile', default='my_results.csv', help='File name to store the predictions')

    args = parser.parse_args()

    config_file = args.config    

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f'ERROR: no such file: {config_file}')
        exit()
        
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # create a new experiment
    expr = Experiment(config)
    util = Util()
    util.debug(f'running {expr.name}, nkululeko version {VERSION}')

    # load the experiment
    expr.load(f'{util.get_save_name()}')
    expr.fill_tests()
    expr.extract_test_feats()
    expr.predict_test_and_save(args.outfile)

    print('DONE')


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd) # use this if you want to state the config file path on command line