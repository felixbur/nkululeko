# main.py
# Demonstration code to use the ML-experiment framework

import argparse
import ast
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from nkululeko.experiment import Experiment
import configparser
from nkululeko.utils.util import Util
from nkululeko.nkululeko import doit as nkulu
from nkululeko.aug_train import doit as aug_train
import nkululeko.glob_conf as glob_conf


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    config = configparser.ConfigParser()
    config.read(config_file)
    datasets = config["EXP"]["databases"]
    datasets = ast.literal_eval(datasets)
    dim = len(datasets)
    results = np.zeros(dim * dim).reshape([dim, dim])

    for i in range(dim):
        for j in range(dim):
            # initialize config
            config = None
            config = configparser.ConfigParser()
            config.read(config_file)
            if i == j:
                dataset = datasets[i]
                print(f"running {dataset}")
                config["DATA"]["databases"] = f"['{dataset}']"
                config["EXP"]["name"] = dataset
            else:
                train = datasets[i]
                test = datasets[j]
                print(f"running train: {train}, test: {test}")
                config["DATA"]["databases"] = f"['{train}', '{test}']"
                config["DATA"][f"{test}.split_strategy"] = "test"
                config["DATA"][f"{train}.split_strategy"] = "train"
                config["EXP"]["name"] = f"{train}_vs_{test}"

            tmp_config = "tmp.ini"
            with open(tmp_config, "w") as tmp_file:
                config.write(tmp_file)
            if config.has_section("AUGMENT"):
                result = aug_train(tmp_config)
            else:
                result = nkulu(tmp_config)
            results[i, j] = float(result)
    print(repr(results))
    root = config["EXP"]["root"]
    plot_name = f"{root}/heatmap.png"
    plot_heatmap(results, datasets, plot_name, config, datasets)


def trunc_to_three(x):
    return int(x * 1000) / 1000.0


def plot_heatmap(results, labels, name, config, datasets):
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
    res_dir = config["EXP"]["root"]
    file_name = f"{res_dir}/results.txt"
    with open(file_name, "w") as text_file:
        text_file.write(
            f"Mean UAR: {mean} (self: {mean_diag}, cross: {mean_non_diag})\n"
        )
        data_s = ", ".join(datasets)
        text_file.write(f"{data_s}\n")
        text_file.write(f"{colsums}\n")

    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap=cm.Blues)
    caption = f"Rows: train, Cols: test. Mean UAR: {mean} (self: {mean_diag}, cross: {mean_non_diag})."
    ax.set_title(caption)
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # sys.argv[1])