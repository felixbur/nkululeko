#!/usr/bin/env python
# -*- coding: utf-8 -*-


import configparser
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util


def ensemble_predictions(config_files, method, no_labels):
    """
    Ensemble predictions from multiple experiments.

    Args:
        config_files (list): List of configuration file paths.
        method (str): Ensemble method to use. Options are 'majority_voting', 'mean', 'max', or 'sum'.
        no_labels (bool): Flag indicating whether the predictions have labels or not.

    Returns:
        pandas.DataFrame: The ensemble predictions.

    Raises:
        ValueError: If an unknown ensemble method is provided.
        AssertionError: If the number of config files is less than 2 for majority voting.

    """
    ensemble_preds = []
    # labels = []
    for config_file in config_files:
        if no_labels:
            # for ensembling results from Nkululeko.demo
            pred = pd.read_csv(config_file)
            labels = pred.columns[1:-2]
        else:
            # for ensembling results from Nkululeko.nkululeko
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
            pred = expr.util.get_pred_name()
            print(f"Loading predictions from {pred}")
            preds = pd.read_csv(pred)

        ensemble_preds.append(preds)

    # pd concate
    ensemble_preds = pd.concat(ensemble_preds, axis=1)

    if method == "majority_voting":
        # majority voting, get mode, works for odd number of models
        # raise error when number of configs only two:
        assert (
            len(config_files) > 2
        ), "Majority voting only works for more than two models"
        ensemble_preds["predicted"] = ensemble_preds.mode(axis=1)[0]

    elif method == "mean":
        for label in labels:
            ensemble_preds[label] = ensemble_preds[label].mean(axis=1)

    elif method == "max":
        for label in labels:
            ensemble_preds[label] = ensemble_preds[label].max(axis=1)
            # get max value from all labels to inver that labels

    elif method == "sum":
        for label in labels:
            ensemble_preds[label] = ensemble_preds[label].sum(axis=1)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # get the highest value from all labels to inver that labels
    # replace the old first predicted column
    ensemble_preds["predicted"] = ensemble_preds[labels].idxmax(axis=1)

    if no_labels:
        return ensemble_preds

    # Drop start, end columns
    ensemble_preds = ensemble_preds.drop(columns=["start", "end"])

    # Drop other column except until truth
    ensemble_preds = ensemble_preds.iloc[:, : len(labels) + 3]

    # calculate UAR from predicted and truth columns

    truth = ensemble_preds["truth"]
    predicted = ensemble_preds["predicted"]
    uar = (truth == predicted).mean()
    Util("ensemble").debug(f"UAR: {uar:.3f}")

    # only return until 'predicted' column
    return ensemble_preds


def main(src_dir):
    parser = ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to the configuration files of the experiments to ensemble. \
             Can be INI files for Nkululeko.nkululeo or CSV files from Nkululeko.demo.",
    )
    parser.add_argument(
        "--method",
        default="majority_voting",
        choices=["majority_voting", "mean", "max", "sum"],
        help="Ensemble method to use (default: majority_voting)",
    )
    parser.add_argument(
        "--outfile",
        default="ensemble_result.csv",
        help="Output file path for the ensemble predictions (default: ensemble_predictions.csv)",
    )

    # add argument if true label is not available
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="True if true labels are not available. For Nkululeko.demo results.",
    )

    args = parser.parse_args()

    start = time.time()

    ensemble_preds = ensemble_predictions(args.configs, args.method, args.no_labels)

    # save to csv
    ensemble_preds.to_csv(args.outfile, index=False)
    print(f"Ensemble predictions saved to: {args.outfile}")
    print(f"Ensemble done, used {time.time()-start:.2f} seconds")

    print("DONE")


if __name__ == "__main__":
    cwd = Path(__file__).parent
    main(cwd)
