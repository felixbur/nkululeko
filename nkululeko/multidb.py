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
    datasets = config["EXP"]["datasets"]
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
                print(f"running {train}_vs_{test}")
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
            # # set up the experiment
            # expr = Experiment(config)
            # util = Util("emodbs")

            # # load the data
            # expr.load_datasets()
            # # split into train and test
            # expr.fill_train_and_tests()
            # util.debug(
            #     f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}"
            # )

            # # extract features
            # expr.extract_feats()
            # util.debug(
            #     f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}"
            # )

            # # initialize a run manager
            # expr.init_runmanager()

            # # run the experiment
            # reports = expr.run()
            # result = reports[-1].result.test
            results[i, j] = float(result)
    print(repr(results))
    root = config["EXP"]["root"]
    plot_name = f"{root}/heatmap.png"
    plot_heatmap(results, datasets, plot_name)


def plot_heatmap(results, labels, name):
    df_cm = pd.DataFrame(
        results, index=[i for i in labels], columns=[i for i in labels]
    )
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap=cm.Blues)
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # sys.argv[1])
