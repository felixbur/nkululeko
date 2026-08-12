"""
Demonstrates the usage of the ML-experiment framework for the nkululeko MULTIDB project.

The `main` function is the entry point of the script, which parses command-line arguments, reads a configuration file, and runs the nkululeko or aug_train functions based on the configuration.

The `plot_heatmap` function generates a heatmap plot of the results and saves it to a file, along with some summary statistics.
"""

# main.py
# Demonstration code to use the ML-experiment framework

import argparse
import ast
import configparser
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from nkululeko.aug_train import doit as aug_train
from nkululeko.nkululeko import doit as nkulu
from nkululeko.utils.errors import NkululukoError


def main(src_dir):
    parser = argparse.ArgumentParser(
        description="Call the nkululeko MULTIDB framework."
    )
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        sys.exit(1)

    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        datasets = config["EXP"]["databases"]
        datasets = ast.literal_eval(datasets)
        try:
            use_splits = eval(config["EXP"]["use_splits"])
        except KeyError:
            use_splits = False
        dim = len(datasets)
        results = np.zeros(dim * dim).reshape([dim, dim])
        last_epochs = np.zeros(dim * dim).reshape([dim, dim])
        # check if some data should be added to training
        try:
            extra_trains = config["CROSSDB"]["train_extra"]
        except KeyError:
            extra_trains = False

        for i in range(dim):
            for j in range(dim):
                # initialize config
                config = None
                config = configparser.ConfigParser()
                config.read(config_file)
                if i == j:
                    dataset = datasets[i]
                    print(f"running {dataset}")
                    if extra_trains:
                        extra_trains_1 = extra_trains.removeprefix("[").removesuffix("]")
                        config["DATA"]["databases"] = f"['{dataset}', {extra_trains_1}]"
                        extra_trains_2 = ast.literal_eval(extra_trains)
                        for extra_train in extra_trains_2:
                            config["DATA"][f"{extra_train}.split_strategy"] = "train"
                    else:
                        config["DATA"]["databases"] = f"['{dataset}']"
                    config["EXP"]["name"] = dataset
                else:
                    train = datasets[i]
                    test = datasets[j]
                    print(f"running train: {train}, test: {test}")
                    if extra_trains:
                        extra_trains_1 = extra_trains.removeprefix("[").removesuffix("]")
                        config["DATA"]["databases"] = (
                            f"['{train}', '{test}', {extra_trains_1}]"
                        )
                        if use_splits:
                            config["DATA"][f"{test}.as_test"] = "True"
                            config["DATA"][f"{train}.as_train"] = "True"
                        else:
                            config["DATA"][f"{test}.split_strategy"] = "test"
                            config["DATA"][f"{train}.split_strategy"] = "train"
                        extra_trains_2 = ast.literal_eval(extra_trains)
                        for extra_train in extra_trains_2:
                            config["DATA"][f"{extra_train}.split_strategy"] = "train"
                    else:
                        config["DATA"]["databases"] = f"['{train}', '{test}']"
                        if use_splits:
                            config["DATA"][f"{test}.as_test"] = "True"
                            config["DATA"][f"{train}.as_train"] = "True"
                        else:
                            config["DATA"][f"{test}.split_strategy"] = "test"
                            config["DATA"][f"{train}.split_strategy"] = "train"
                    config["EXP"]["name"] = f"{train}_vs_{test}"

                tmp_config = "tmp.ini"
                with open(tmp_config, "w") as tmp_file:
                    config.write(tmp_file)
                if config.has_section("AUGMENT"):
                    result, last_epoch = aug_train(tmp_config)
                else:
                    result, last_epoch = nkulu(tmp_config)
                results[i, j] = float(result)
                last_epochs[i, j] = int(last_epoch)
        print(repr(results))
        print(repr(last_epochs))
        root = os.path.join(config["EXP"]["root"], "")
        try:
            format = config["PLOT"]["format"]
            plot_name = f"{root}/heatmap.{format}"
        except KeyError:
            plot_name = f"{root}/heatmap.png"
        plot_heatmap(results, last_epochs, datasets, plot_name, config, datasets)
    except NkululukoError as e:
        print(str(e))
        sys.exit(1)


def trunc_to_three(x):
    return int(x * 1000) / 1000.0


def _metric_label(config):
    """Return the display name for the metric in `results`.

    Mirrors Util.high_is_good()'s defaulting: MODEL.measure defaults to
    "uar" for classification experiments and "mse" for regression ones, so
    the heatmap/results.txt label the actual configured metric instead of
    always saying "UAR" (which is meaningless for a regression run).
    """
    if config.has_section("EXP"):
        exp_type = config["EXP"].get("type", "classification")
    else:
        exp_type = "classification"
    default_measure = "uar" if exp_type == "classification" else "mse"
    measure = (
        config["MODEL"].get("measure", default_measure)
        if config.has_section("MODEL")
        else default_measure
    )
    return measure.upper()


def plot_heatmap(results, last_epochs, labels, name, config, datasets):
    df_cm = pd.DataFrame(
        results, index=[i for i in labels], columns=[i for i in labels]
    )
    mean = trunc_to_three(results.mean())
    mean_diag = trunc_to_three(results.diagonal().mean())
    mean_non_diag = trunc_to_three(
        (results.sum() - results.diagonal().sum())
        / (results.size - results.diagonal().size)
    )
    colsums = results.mean(axis=0)
    vfunc = np.vectorize(trunc_to_three)
    colsums = vfunc(colsums)
    rowsums = results.mean(axis=1)
    rowsums = vfunc(rowsums)
    colsums_epochs = last_epochs.mean(axis=0)
    colsums_epochs = vfunc(colsums_epochs)
    metric = _metric_label(config)
    res_dir = config["EXP"]["root"]
    file_name = f"{res_dir}/results.txt"
    with open(file_name, "w") as text_file:
        text_file.write(
            f"Mean {metric}: {mean} (self: {mean_diag}, cross: {mean_non_diag})\n"
        )
        data_s = ", ".join(datasets)
        text_file.write(f"{data_s}\n")
        colsums = np.array2string(colsums, separator=", ")
        text_file.write(f"column means\n{colsums}\n")
        rowsums = np.array2string(rowsums, separator=", ")
        text_file.write(f"rows means\n{rowsums}\n")
        text_file.write("all results\n")
        text_file.write(repr(results))
        text_file.write("\n")
        colsums_epochs = np.array2string(colsums_epochs, separator=", ")
        text_file.write(f"column sums epochs\n{colsums_epochs}\n")
        text_file.write("all epochs\n")
        text_file.write(repr(last_epochs))

    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap=cm.Blues)
    caption = f"Rows: train, Cols: test. Mean {metric}: {mean} (self: {mean_diag}, cross: {mean_non_diag})."
    ax.set_title(caption)
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # sys.argv[1])
