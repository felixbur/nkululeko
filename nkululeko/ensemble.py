#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import configparser
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util

import torch
import logging

# Constants
DEFAULT_METHOD = "majority_voting"
DEFAULT_OUTFILE = "ensemble_result.csv"
COLUMN_PREDICTED = "predicted"

# Setup logging
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

# Setup logging with custom formatter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)


def ensemble_predictions(config_files: List[str], method: str, no_labels: bool) -> pd.DataFrame:
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
    ensemble_preds_ls = []
    for config_file in config_files:
        if no_labels:
            # for ensembling results from Nkululeko.demo
            preds = pd.read_csv(config_file)
            labels = preds.columns[1:-2]
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
            pred_name = expr.util.get_pred_name()
            print(f"Loading predictions from {pred_name}")
            preds = pd.read_csv(pred_name)
            
        ensemble_preds_ls.append(preds)

    # pd concate
    ensemble_preds = pd.concat(ensemble_preds_ls, axis=1)

    if method == "majority_voting":
        """Majority voting, get mode, works for odd number of models
        raise error when number of configs only two"""
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

    elif method == "uncertainty":
        final_predictions = []
        best_uncertainty = []
        for _, row in ensemble_preds.iterrows():
            uncertainties = row[['uncertainty']].values
            max_uncertainty_idx = np.argmax(uncertainties)
            final_predictions.append(row['predicted'].iloc[max_uncertainty_idx])
            best_uncertainty.append(uncertainties[max_uncertainty_idx])

        ensemble_preds['predicted'] = final_predictions
        ensemble_preds['uncertainty'] = best_uncertainty
        # print(f"uncertainty result: {ensemble_preds.head()}")
    
    elif method == "max_class":
        """Get the class with the highest probability across all models."""
        final_preds = []
        final_probs = []
        
        for _, row in ensemble_preds.iterrows():
            max_probs = []
            max_classes = []
            
            for model_df in ensemble_preds_ls:
                model_probs = row[labels].astype(float)
                max_prob = model_probs.max()
                max_class = model_probs.idxmax()
                
                max_probs.append(max_prob)
                max_classes.append(max_class)
            
            # Find the model with the highest max probability
            best_model_index = np.argmax(max_probs)
            
            final_preds.append(max_classes[best_model_index])
            final_probs.append(max_probs[best_model_index])
        
        ensemble_preds['predicted'] = final_preds
        # ensemble_preds['max_probability'] = final_probs

    
    elif method == "entropy":
        """Get the class with the lowest entropy across all models.
        entropy is calculated from confidence score of each prediction, 
        which is calculated using softmax from proability of each class."""
        from scipy.stats import entropy

        final_predictions = []
        final_confidence_scores = []
        
        for idx, _ in ensemble_preds.iterrows():
            model_probas = []
            model_preds = []
            
            for model_df in ensemble_preds_ls:
                model_row = model_df.loc[idx]
                # calculate confidence score for each row using softmax
                probas_sof = torch.nn.functional.softmax(torch.tensor(model_row[labels].values.astype(float)), dim=0)
                model_probas.append(probas_sof)
                model_preds.append(model_row['predicted'])
            
            # Calculate entropy for each model's prediction
            entropies = [entropy(proba) for proba in model_probas]
            
            # Select the model with the lowest entropy
            best_model_idx = np.argmin(entropies)
            
            final_predictions.append(model_preds[best_model_idx])
            final_confidence_scores.append(model_probas[best_model_idx])

        ensemble_preds['predicted'] = final_predictions
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # get the highest value from all labels to inver that labels
    # replace the old first predicted column
    if method in ["mean", "max", "sum"]:
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
    uar = balanced_accuracy_score(truth, predicted)
    acc = (truth == predicted).mean()
    Util("ensemble").debug(f"UAR: {uar:.3f}, ACC: {acc:.3f}")

    # only return until 'predicted' column
    return ensemble_preds


def main(src_dir: Path) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to the configuration files of the experiments to ensemble. \
             Can be INI files for Nkululeko.nkululeo or CSV files from Nkululeko.demo.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["majority_voting", "mean", "max", "sum", "uncertainty", "entropy", "max_class"],
        help=f"Ensemble method to use (default: {DEFAULT_METHOD})",
    )
    parser.add_argument(
        "--outfile",
        default=DEFAULT_OUTFILE,
        help=f"Output file path for the ensemble predictions (default: {DEFAULT_OUTFILE})",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="True if true labels are not available. For Nkululeko.demo results.",
    )

    args = parser.parse_args()

    start = time.time()

    try:
        ensemble_preds = ensemble_predictions(args.configs, args.method, args.no_labels)

        # save to csv
        ensemble_preds.to_csv(args.outfile, index=False)
        logger.info(f"Ensemble predictions saved to: {args.outfile}")
        logger.info(f"Ensemble done, used {time.time()-start:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred during ensemble prediction: {str(e)}")
        return

    logger.info("DONE")

if __name__ == "__main__":
    cwd = Path(__file__).parent
    main(cwd)