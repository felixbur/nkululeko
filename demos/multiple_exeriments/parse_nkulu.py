
import sys
sys.path.append("../../src")
import constants
import numpy as np
import experiment as exp
import configparser
from util import Util
import argparse
import os.path


def main():
    parser = argparse.ArgumentParser(description='Call the nkululeko framework.')
    parser.add_argument('--data', help='The databases', nargs='*', \
        action='append')
    parser.add_argument('--label', nargs='*', help='The labels for the target', \
        action='append')
    parser.add_argument('--tuning_params', nargs='*', help='parameters to be tuned', \
        action='append')
    parser.add_argument('--model', default='xgb', help='The model type', required=True)
    parser.add_argument('--feat', default='os', help='The model type')
    parser.add_argument('--set', help='The opensmile set')
    parser.add_argument('--with_os', help='To add os features')
    parser.add_argument('--target', help='The target designation')
    
    args = parser.parse_args()

    config_file = './exp.ini'
    util = Util()
    # test if config is there
    if not os.path.isfile(config_file):
        util.error(f'no such file {config_file}')

    config = configparser.ConfigParser()
    config.read(config_file)
    # fill the config


    if args.data is not None:
        databases = []
        for t in args.data:
            databases.append(t[0])
        print(f'got databases: {databases}')
        config['DATA']['databases'] = str(databases)
    if args.label is not None:
        labels = []
        for l in args.label:
            labels.append(l[0])
        print(f'got labels: {labels}')
        config['DATA']['labels'] = str(labels)
    if args.tuning_params is not None:
        tuning_params = []
        for tp in args.tuning_params:
            tuning_params.append(tp[0])
        config['MODEL']['tuning_params'] = str(tuning_params)
    if args.target is not None:
        config['DATA']['target'] = args.target
    if args.model is not None:
        config['MODEL']['type'] = args.model
    if args.feat is not None:
        config['FEATS']['type'] = args.feat
    if args.with_os is not None:
        config['FEATS']['with_os'] = args.with_os
    if args.set is not None:
        config['FEATS']['set'] = args.set


    name = config['EXP']['name']
    util = Util()
    util.debug(f'running {name}, Nkululeko version {constants.VERSION}')

    # init the experiment
    expr = exp.Experiment(config)
    # load the data
    expr.load_datasets()
    # split into train and test
    expr.fill_train_and_tests()
    # extract features
    expr.extract_feats()
    # initialize a run manager
    expr.init_runmanager()
    # run the experiment
    reports = expr.run()
    result = reports[-1].result.test
    # report result
    util.debug(f'result for {expr.get_name()} is {result}')

if __name__ == "__main__":
    main()