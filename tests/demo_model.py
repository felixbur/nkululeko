# main.py
# Demonstration code to use the ML-experiment framework
# Test the loading of a previously trained model and demo mode
# needs the project tests/exp_emodb to run before

import sys
sys.path.append("./src")
import experiment as exp
import configparser
from util import Util

def main(config_file):
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    util = Util()
    
    # create a new experiment
    expr = exp.Experiment(config)
    print(f'running {expr.name}')

    # load the experiment
    expr.load(f'{util.get_save_name()}')

    expr.demo()

    print('DONE')


if __name__ == "__main__":
    main('tests/exp_emodb.ini')
