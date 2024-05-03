# test_pretrain.py
import argparse
import configparser
import os.path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers

import audeer
import audiofile

from nkululeko.constants import VERSION
import nkululeko.experiment as exp
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


def doit(config_file):
    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    module = "test_pretrain"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    sampling_rate = 16000
    max_duration_sec = 8.0

    model_path = "facebook/wav2vec2-large-robust-ft-swbd-300h"
    num_layers = None

    batch_size = 16
    accumulation_steps = 4

    # create dataset

    dataset = {}
    data_sources = {
        "train": pd.DataFrame(expr.df_train[glob_conf.target]),
        "dev": pd.DataFrame(expr.df_test[glob_conf.target]),
    }

    for split in ["train", "dev"]:

        y = pd.Series(
            data=data_sources[split].itertuples(index=False, name=None),
            index=data_sources[split].index,
            dtype=object,
            name="labels",
        )

        y.name = "targets"
        df = y.reset_index()
        df.start = df.start.dt.total_seconds()
        df.end = df.end.dt.total_seconds()
        print(f"{split}: {len(df)}")
        ds = datasets.Dataset.from_pandas(df)
        dataset[split] = ds

        dataset = datasets.DatasetDict(dataset)

    config = transformers.AutoConfig.from_pretrained(
        model_path,
        num_labels=len(util.la),
        label2id=data.gender_mapping,
        id2label=data.gender_mapping_reverse,
        finetuning_task="age-gender",
    )
    if num_layers is not None:
        config.num_hidden_layers = num_layers
    setattr(config, "sampling_rate", sampling_rate)
    setattr(config, "data", ",".join(sources))

    print("DONE")


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"
    doit(config_file)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
