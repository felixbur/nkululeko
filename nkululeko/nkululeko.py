#!/usr/bin/env python3

# nkululeko.py: Entry script to do a Nkululeko experiment

import argparse
import ast
import configparser
import os
from pathlib import Path

import numpy as np

import nkululeko.experiment as exp
import nkululeko.glob_conf as glob_conf
from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


def doit(config_file):
    # test if the configuration file exists
    if not Path(config_file).is_file():
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    module = "nkululeko"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    # When DATA.tests is provided and a saved experiment already exists, skip
    # training and evaluate the stored best model on the new test set instead.
    has_tests = util.config_val("DATA", "tests", False)
    save_name = util.get_save_name()
    if has_tests and os.path.isfile(save_name):
        util.debug(
            f"DATA.tests is set and saved experiment found at {save_name}"
            " — loading best model, skipping training"
        )
        expr.load(save_name)
        # Restore the label encoder in the global namespace so the Reporter
        # can map integer class indices back to string label names.
        glob_conf.set_label_encoder(expr.label_encoder)

        # Load test data with original string labels (no encoding).
        expr.fill_tests(encode=False)
        expr.extract_test_feats()

        best_model = expr.runmgr.get_best_model()

        # Integer-encode the target column in-place so model.predict() can
        # compare predictions against numeric ground-truth values.
        expr.df_test[expr.target] = expr.label_encoder.transform(
            expr.df_test[expr.target]
        )
        best_model.reset_test(expr.df_test, expr.feats_test)
        report = best_model.predict()
        report.set_id(best_model.run, best_model.epoch)

        # Print classification report and save confusion matrix plot.
        test_dbs_str = "-".join(ast.literal_eval(has_tests))
        plot_name = (
            util.get_exp_name()
            + f"_{test_dbs_str}_{best_model.run}_{best_model.epoch:03d}_cnf"
        )
        report.print_results(best_model.epoch, file_name=plot_name)
        report.plot_confmatrix(plot_name, best_model.epoch)

        result = report.result.test
        print("DONE")
        return result, best_model.epoch

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()

    # extract features
    expr.extract_feats()

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    reports, last_epochs = expr.run()
    result = expr.get_best_report(reports).result.test

    # evaluate per test dataset when multiple test sets are present
    expr.evaluate_per_test_set()

    expr.store_report()

    # check if we want to export the model
    o_path = util.config_val("EXP", "export_onnx", "False")
    if o_path.lower() in ["true", "1", "yes"]:
        util.info(f"Exporting ONNX model to {o_path}")
        o_path = o_path.replace('"', "")
        expr.runmgr.get_best_model().export_onnx(str(o_path))

    print("DONE")
    return result, int(np.asarray(last_epochs).min())


def main():
    cwd = Path(__file__).parent.absolute()
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--version", action="version", version=f"Nkululeko {VERSION}")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = cwd / "exp.ini"
    doit(config_file)


if __name__ == "__main__":
    main()  # use this if you want to state the config file path on command line
