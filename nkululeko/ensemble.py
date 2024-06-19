#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ensemble.py: combining multiple Nkululeko results
# supported ensemble methods: majority_voting, mean, max, sum
# for majority_voting: use output directly (without --outfile)
# for mean, max, sum: use ~~demo~~ results from previous experiments

from ast import arg
from json import load
from logging import config
import os
from argparse import ArgumentParser
import configparser
from dataclasses_json import global_config
from librosa import ex
import pandas as pd
import numpy as np
from sklearn import ensemble

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util
from nkululeko.test_predictor import TestPredictor
from nkululeko.demo_predictor import Demo_predictor

import time

def ensemble_predictions(config_files, method):
    """Ensemble predictions from multiple experiments."""
    
    ensemble_preds = []
    # labels = []
    for config_file in config_files:
        # create a new experiment
        config = configparser.ConfigParser()
        config.read(config_file)
        expr = Experiment(config)
        module = "ensemble"
        expr.set_module(module)
        util = Util(module, has_config=True)
        util.debug(
            f"running {expr.name} from config {config_file}, nkululeko version"
            f" {VERSION}"
        )

        # get labels
        labels = expr.util.get_labels()
        # load the experiment
        # get CSV files of predictions
        pred = expr.util.get_save_name_csv()
        print(f"Loading predictions from {pred}")
        preds = pd.read_csv(pred)
        ensemble_preds.append(preds)

    # pd concate
    ensemble_preds = pd.concat(ensemble_preds, axis=1)

    if method == 'majority_voting':
        # majority voting, get mode, works for odd number of models
        # raise error when number of configs only two:
        assert len(config_files) > 2, "Majority voting only works for more than two models"
        ensemble_preds['predicted'] = ensemble_preds.mode(axis=1)[0]

    elif method == 'mean':
        for label in labels:
            ensemble_preds[label] = ensemble_preds[label].mean(axis=1)

    elif method == 'max':
        for label in labels:
            ensemble_preds[label] = ensemble_preds[label].max(axis=1)

    elif method == 'sum':
        for label in labels:
            ensemble_preds[label] = ensemble_preds[label].sum(axis=1)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # only return until 'predicted' column
    return ensemble_preds.iloc[:, :len(labels) + 2] # labels + index + predict
    
def main():
    parser = ArgumentParser()
    parser.add_argument(
        'configs',
        nargs='+',
        help='Paths to the configuration files of the experiments to ensemble')
    parser.add_argument(
        '--method',
        default='majority_voting',
        choices=['majority_voting', 'mean', 'max'],
        help='Ensemble method to use (default: majority_voting)')
    parser.add_argument(
        '--outfile',
        default='ensemble_result.csv',
        help='Output file path for the ensemble predictions (default: ensemble_predictions.csv)')
    
    args = parser.parse_args()
      
    start = time.time()
    # for majority voting

    ensemble_preds = ensemble_predictions(args.configs, args.method)
    
    # save to csv
    ensemble_preds.to_csv(args.outfile, index=False)
    print(f"Ensemble predictions saved to: {args.outfile}")
    print(f"Ensemble done, used {time.time()-start:.2f} seconds")

    print("DONE")

if __name__ == '__main__':
    main()
