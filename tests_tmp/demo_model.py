# main.py
# Demonstration code to use the ML-experiment framework
# Test the loading of a previously trained model and demo mode
# needs the project tests/exp_emodb to run before

import sys
sys.path.append("./src")
from nkululeko.experiment import Experiment
import configparser
from nkululeko.util import Util
from  nkululeko.constants import VERSION

def main(config_file):
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # create a new experiment
    expr = Experiment(config)
    util = Util()
    util.debug(f'running {expr.name}, nkululeko version {VERSION}')

    # load the experiment
    expr.load(f'{util.get_save_name()}')

    expr.demo()

    print('DONE')


if __name__ == "__main__":
    main('tests/exp_emodb.ini')
