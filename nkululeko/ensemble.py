#!/usr/bin/env python3
"""
Ensemble predictions from multiple experiments.

Args:
    config_files (list): List of configuration file paths.
    method (str): Ensemble method to use. Options are 'majority_voting', 'mean', 'max', 'sum', 'uncertainty', 'uncertainty_weighted', 'confidence_weighted', or 'performance_weighted'.
    threshold (float): Threshold for the 'uncertainty' ensemble method (default: 1.0, i.e. no threshold).
    weights (list): Weights for the 'performance_weighted' ensemble method.
    no_labels (bool): Flag indicating whether the predictions have labels or not.

Returns:
    pandas.DataFrame: The ensemble predictions.

Raises:
    ValueError: If an unknown ensemble method is provided.
    AssertionError: If the number of config files is less than 2 for majority voting.
"""


import configparser
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util

# import torch

# Constants
DEFAULT_METHOD = "mean"
DEFAULT_OUTFILE = "ensemble_result.csv"


def majority_voting(ensemble_preds_ls):
    all_predictions = pd.concat([df["predicted"] for df in ensemble_preds_ls], axis=1)
    return all_predictions.mode(axis=1).iloc[:, 0]


def mean_ensemble(ensemble_preds, labels):
    for label in labels:
        ensemble_preds[label] = ensemble_preds[label].mean(axis=1)
    return ensemble_preds[labels].idxmax(axis=1)


def max_ensemble(ensemble_preds, labels):
    for label in labels:
        ensemble_preds[label] = ensemble_preds[label].max(axis=1)
    return ensemble_preds[labels].idxmax(axis=1)


def sum_ensemble(ensemble_preds, labels):
    for label in labels:
        ensemble_preds[label] = ensemble_preds[label].sum(axis=1)
    return ensemble_preds[labels].idxmax(axis=1)


def uncertainty_ensemble(ensemble_preds_ls, labels, threshold):
    final_predictions = []
    final_uncertainties = []

    for idx in ensemble_preds_ls[0].index:
        uncertainties = [df.loc[idx, "uncertainty"] for df in ensemble_preds_ls]
        min_uncertainty_idx = np.argmin(uncertainties)
        min_uncertainty = uncertainties[min_uncertainty_idx]

        if min_uncertainty <= threshold:
            # Use the prediction with low uncertainty
            final_predictions.append(
                ensemble_preds_ls[min_uncertainty_idx].loc[idx, "predicted"]
            )
            final_uncertainties.append(min_uncertainty)
        else:  # for uncertainty above threshold
            # Calculate mean of probabilities same class different model
            mean_probs = np.mean(
                [df.loc[idx, labels].values for df in ensemble_preds_ls], axis=0
            )
            final_predictions.append(labels[np.argmax(mean_probs)])
            final_uncertainties.append(np.mean(uncertainties))

    return final_predictions


def uncertainty_weighted_ensemble(ensemble_preds_ls, labels):
    """Weighted ensemble based on uncertainty, normalized for each class"""
    final_predictions = []
    final_uncertainties = []

    for idx in ensemble_preds_ls[0].index:
        uncertainties = [df.loc[idx, "uncertainty"] for df in ensemble_preds_ls]
        # Convert uncertainties to accuracies/confidence
        accuracies = [1 - uncertainty for uncertainty in uncertainties]

        # Calculate weights (inverse of uncertainties)
        weights = [
            1 / uncertainty if uncertainty != 0 else 1e10
            for uncertainty in uncertainties
        ]

        # Normalize weights for each class
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted probabilities for each class
        weighted_probs = {label: 0 for label in labels}
        for df, weight in zip(ensemble_preds_ls, normalized_weights):
            for label in labels:
                weighted_probs[label] += df.loc[idx, label] * weight

        # Select the class with the highest weighted probability
        predicted_class = max(weighted_probs, key=weighted_probs.get)
        final_predictions.append(predicted_class)

        # Use the lowest accuracy as the final uncertainty
        final_uncertainties.append(1 - min(accuracies))

    return final_predictions, final_uncertainties


def confidence_weighted_ensemble(ensemble_preds_ls, labels):
    """Weighted ensemble based on confidence, normalized for all samples per model"""
    final_predictions = []
    final_confidences = []

    for idx in ensemble_preds_ls[0].index:
        class_probabilities = {label: 0 for label in labels}
        total_confidence = 0

        for df in ensemble_preds_ls:
            row = df.loc[idx]
            confidence = 1 - row["uncertainty"]  # confidence score
            total_confidence += confidence

            for label in labels:
                class_probabilities[label] += row[label] * confidence

        # Normalize probabilities
        for label in labels:
            class_probabilities[label] /= total_confidence

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        final_predictions.append(predicted_class)
        final_confidences.append(max(class_probabilities.values()))

    return final_predictions, final_confidences


def performance_weighted_ensemble(ensemble_preds_ls, labels, weights):
    """Weighted ensemble based on performances"""
    final_predictions = []
    final_confidences = []

    # asserts weiths in decimal 0-1
    assert all(0 <= w <= 1 for w in weights), "Weights must be between 0 and 1"

    # assert lenght of weights matches number of models
    assert len(weights) == len(
        ensemble_preds_ls
    ), "Number of weights must match number of models"

    # Normalize weights
    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]

    for idx in ensemble_preds_ls[0].index:
        class_probabilities = {label: 0 for label in labels}

        for df, weight in zip(ensemble_preds_ls, weights):
            row = df.loc[idx]
            for label in labels:
                class_probabilities[label] += row[label] * weight

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        final_predictions.append(predicted_class)
        final_confidences.append(max(class_probabilities.values()))

    return final_predictions, final_confidences


def ensemble_predictions(
    config_files: List[str],
    method: str,
    threshold: float,
    weights: List[float],
    no_labels: bool,
) -> pd.DataFrame:
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
            util.debug(f"Loading predictions from {pred_name}")
            preds = pd.read_csv(pred_name)

        ensemble_preds_ls.append(preds)

    # pd concate
    ensemble_preds = pd.concat(ensemble_preds_ls, axis=1)

    if method == "majority_voting":
        assert (
            len(ensemble_preds_ls) > 2
        ), "Majority voting only works for more than two models"
        ensemble_preds["predicted"] = majority_voting(ensemble_preds_ls)
    elif method == "mean":
        ensemble_preds["predicted"] = mean_ensemble(ensemble_preds, labels)
    elif method == "max":
        ensemble_preds["predicted"] = max_ensemble(ensemble_preds, labels)
    elif method == "sum":
        ensemble_preds["predicted"] = sum_ensemble(ensemble_preds, labels)
    elif method == "uncertainty":
        ensemble_preds["predicted"] = uncertainty_ensemble(
            ensemble_preds_ls, labels, threshold
        )
    elif method == "uncertainty_weighted":
        (
            ensemble_preds["predicted"],
            ensemble_preds["uncertainty"],
        ) = uncertainty_weighted_ensemble(ensemble_preds_ls, labels)
    elif method == "confidence_weighted":
        (
            ensemble_preds["predicted"],
            ensemble_preds["confidence"],
        ) = confidence_weighted_ensemble(ensemble_preds_ls, labels)
    elif method == "performance_weighted":
        (
            ensemble_preds["predicted"],
            ensemble_preds["confidence"],
        ) = performance_weighted_ensemble(ensemble_preds_ls, labels, weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # get the highest value from all labels to infer the label
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
    # print classification report
    Util("ensemble").debug(f"\n {classification_report(truth, predicted, digits=4)}")
    Util("ensemble").debug(f"{method}: UAR: {uar:.3f}, ACC: {acc:.3f}")

    return ensemble_preds


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="+",
        help="Paths to the configuration files of the experiments to ensemble. \
             Can be INI files for Nkululeko.nkululeko or CSV files from Nkululeko.demo.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=[
            "majority_voting",
            "mean",
            "max",
            "sum",
            # "max_class",
            # "uncertainty_lowest",
            # "entropy",
            "uncertainty",
            "uncertainty_weighted",
            "confidence_weighted",
            "performance_weighted",
        ],
        help=f"Ensemble method to use (default: {DEFAULT_METHOD})",
    )
    # add threshold if method is uncertainty_threshold
    parser.add_argument(
        "--threshold",
        default=1.0,
        type=float,
        help="Threshold for uncertainty_threshold method (default: 1.0, i.e. no threshold)",
    )
    parser.add_argument(
        "--outfile",
        default=DEFAULT_OUTFILE,
        help=f"Output file path for the ensemble predictions (default: {DEFAULT_OUTFILE})",
    )
    parser.add_argument(
        "--weights",
        default=None,
        nargs="+",
        type=float,
        help="Weights for the ensemble method (default: None, e.g. 0.5 0.5)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="True if true labels are not available. For Nkululeko.demo results.",
    )

    args = parser.parse_args()

    start = time.time()

    ensemble_preds = ensemble_predictions(
        args.config, args.method, args.threshold, args.weights, args.no_labels
    )

    # save to csv
    ensemble_preds.to_csv(args.outfile, index=False)
    Util("ensemble").debug(f"Ensemble predictions saved to: {args.outfile}")
    Util("ensemble").debug(f"Ensemble done, used {time.time()-start:.2f} seconds")

    Util("ensemble").debug("DONE")


if __name__ == "__main__":
    main()
