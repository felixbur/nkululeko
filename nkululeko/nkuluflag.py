import argparse
import configparser
import os
import os.path
import sys

from nkululeko.nkululeko import doit as nkulu
from nkululeko.test import do_it as test_mod


def doit(cla):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", help="The base configuration")
    parser.add_argument("--mod", default="nkulu", help="Which nkululeko module to call")
    parser.add_argument("--data", help="The databases", nargs="*", action="append")
    parser.add_argument(
        "--label", nargs="*", help="The labels for the target", action="append"
    )
    parser.add_argument(
        "--tuning_params", nargs="*", help="parameters to be tuned", action="append"
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        help="layer config for mlp, e.g. l1:128 ",
        action="append",
    )
    parser.add_argument("--model", default="xgb", help="The model type")
    parser.add_argument("--feat", default="['os']", help="The feature type")
    parser.add_argument("--set", help="The opensmile set")
    parser.add_argument("--target", help="The target designation")
    parser.add_argument("--epochs", help="The number of epochs")
    parser.add_argument("--runs", help="The number of runs")
    parser.add_argument("--learning_rate", help="The learning rate")
    parser.add_argument("--drop", help="The dropout rate [0:1]")

    args = parser.parse_args(cla)

    if args.config is not None:
        config_file = args.config
    else:
        print("ERROR: need config file")
        quit(-1)

    if args.mod is not None:
        nkulu_mod = args.mod

    # test if config is there
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file {config_file}")

    config = configparser.ConfigParser()
    config.read(config_file)
    # fill the config

    if args.data is not None:
        databases = []
        for t in args.data:
            databases.append(t[0])
        print(f"got databases: {databases}")
        config["DATA"]["databases"] = str(databases)
    if args.label is not None:
        labels = []
        for label in args.label:
            labels.append(label[0])
        print(f"got labels: {labels}")
        config["DATA"]["labels"] = str(labels)
    if args.tuning_params is not None:
        tuning_params = []
        for tp in args.tuning_params:
            tuning_params.append(tp[0])
        config["MODEL"]["tuning_params"] = str(tuning_params)
    if args.layers is not None:
        config["MODEL"]["layers"] = args.layers[0][0]
    if args.target is not None:
        config["DATA"]["target"] = args.target
    if args.epochs is not None:
        config["EXP"]["epochs"] = args.epochs
    if args.runs is not None:
        config["EXP"]["runs"] = args.runs
    if args.learning_rate is not None:
        config["MODEL"]["learning_rate"] = args.learning_rate
    if args.drop is not None:
        config["MODEL"]["drop"] = args.drop
    if args.model is not None:
        config["MODEL"]["type"] = args.model
    if args.feat is not None:
        config["FEATS"]["type"] = f"['{args.feat}']"
    if args.set is not None:
        config["FEATS"]["set"] = args.set
    tmp_config = "tmp.ini"
    with open(tmp_config, "w") as tmp_file:
        config.write(tmp_file)

    result, last_epoch = 0, 0
    if nkulu_mod == "nkulu":
        result, last_epoch = nkulu(tmp_config)
    elif nkulu_mod == "test":
        result, last_epoch = test_mod(tmp_config, "test_results.csv")
    else:
        print(f"ERROR: unknown module: {nkulu_mod}, should be [nkulu | test]")
    return result, last_epoch


if __name__ == "__main__":
    cla = sys.argv
    cla.pop(0)
    doit(cla)  # sys.argv[1])
