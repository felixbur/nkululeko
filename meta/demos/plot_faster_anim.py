# main.py
# Demonstration code to use the ML-experiment framework

import sys
sys.path.append("./src")
from reporter import Reporter
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
    # expr.load(f'{util.get_exp_name()}.pkl')

    reporter = Reporter([], [])

    reporter.make_conf_animation('test.gif')

    print('DONE')


if __name__ == "__main__":
    main(sys.argv[1])
