
import sys
#sys.path.append("../../src")
sys.path.append("../nkululeko/src")
import constants
import numpy as np
import experiment as exp
import configparser
from util import Util
import argparse
import os.path
src_path = './demos/multiple_experiments/'


def main():
    parser = argparse.ArgumentParser(description='Call the nkululeko framework.')
    parser.add_argument('--config', default='exp.ini', help='The base configuration')
    parser.add_argument('--data', help='The databases', nargs='*', \
        action='append')
    parser.add_argument('--label', nargs='*', help='The labels for the target', \
        action='append')
    parser.add_argument('--tuning_params', nargs='*', help='parameters to be tuned', \
        action='append')
    parser.add_argument('--layers', nargs='*', help='layer config for mlp, e.g. l1:128 ', \
        action='append')
    parser.add_argument('--model', default='xgb', help='The model type')
    parser.add_argument('--feat', default='os', help='The feature type')
    parser.add_argument('--set', help='The opensmile set')
    parser.add_argument('--with_os', help='To add os features')
    parser.add_argument('--target', help='The target designation')
    parser.add_argument('--epochs', help='The number of epochs')
    parser.add_argument('--runs', help='The number of runs')
    parser.add_argument('--learning_rate', help='The learning rate')
    parser.add_argument('--drop', help='The dropout rate [0:1]')
    
    args = parser.parse_args()

    if args.config is not None:
        config_file = args.config    
    else:
        config_file = f'{src_path}exp.ini'
    # test if config is there
    if not os.path.isfile(config_file):
        print(f'ERROR: no such file {config_file}')

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
    if args.layers is not None:
        config['MODEL']['layers'] = args.layers[0][0]
    if args.target is not None:
        config['DATA']['target'] = args.target
    if args.epochs is not None:
        config['EXP']['epochs'] = args.epochs
    if args.runs is not None:
        config['EXP']['runs'] = args.runs
    if args.learning_rate is not None:
        config['MODEL']['learning_rate'] = args.learning_rate
    if args.drop is not None:
        config['MODEL']['drop'] = args.drop
    if args.model is not None:
        config['MODEL']['type'] = args.model
    if args.feat is not None:
        config['FEATS']['type'] = f'[\'{args.feat}\']'
    if args.with_os is not None:
        config['FEATS']['with_os'] = args.with_os
    if args.set is not None:
        config['FEATS']['set'] = args.set


    name = config['EXP']['name']

    # init the experiment
    expr = exp.Experiment(config)
    util = Util()
    util.debug(f'running {name}, Nkululeko version {constants.VERSION}')

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