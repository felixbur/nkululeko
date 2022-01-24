# main.py
# Demonstration code to use the ML-experiment framework
# Test the loading of a previously trained model and save the results of the test data as a new database
# needs the project tests/exp_emodb to run before

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
    util = Util()
    
    # create a new experiment
    expr = exp.Experiment(config)
    print(f'running {expr.name}, nkululeko version {constants.VERSION}')

    # load the experiment
    expr.load(f'{util.get_save_name()}')

    expr.predict_test_and_save('predict_and_store_test')

    print('DONE')


if __name__ == "__main__":
    main('tests/exp_cross.ini')
