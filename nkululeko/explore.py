"""
Explore the feature sets of a machine learning experiment.

This script is the entry point for the 'explore' module of the nkululeko framework.
It handles loading the experiment configuration, setting up the experiment, and
running various feature exploration techniques based on the configuration.

The script supports the following configuration options:
- `no_warnings`: If set to `True`, it will ignore all warnings during the exploration.
- `feature_distributions`: If set to `True`, it will generate plots of the feature distributions.
- `tsne`: If set to `True`, it will generate a t-SNE plot of the feature space.
- `scatter`: If set to `True`, it will generate a scatter plot of the feature space.
- `spotlight`: If set to `True`, it will generate a 'spotlight' plot of the feature space.
- `shap`: If set to `True`, it will generate SHAP feature importance plots.
- `model`: The type of model to use for the feature exploration (e.g. 'SVM').
- `plot_tree`: If set to `True`, it will generate a decision tree plot.

The script can be run from the command line with the `--config` argument to specify
the configuration file to use. If no configuration file is provided, it will look
for an `exp.ini` file in the same directory as the script.
"""

# explore.py
# explore the feature sets

import argparse
import configparser
from pathlib import Path

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util


def main():
    parser = argparse.ArgumentParser(
        description="Call the nkululeko EXPLORE framework."
    )
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    config_file = args.config if args.config is not None else "exp.ini"

    if not Path(config_file).is_file():
        print(f"ERROR: no such file: {config_file}")
        exit()

    config = configparser.ConfigParser()
    config.read(config_file)
    expr = Experiment(config)
    module = "explore"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")
    needs_feats = False
    try:
        # load the experiment
        expr.load(f"{util.get_save_name()}")
        needs_feats = True
    except FileNotFoundError:
        # first time: load the data
        expr.load_datasets()

        # split into train and test
        expr.fill_train_and_tests()
        util.debug(
            f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}"
        )

        plot_feats = eval(util.config_val("EXPL", "feature_distributions", "False"))
        tsne = eval(util.config_val("EXPL", "tsne", "False"))
        scatter = eval(util.config_val("EXPL", "scatter", "False"))
        spotlight = eval(util.config_val("EXPL", "spotlight", "False"))
        shap = eval(util.config_val("EXPL", "shap", "False"))
        model_type = util.config_val("EXPL", "model", False)
        plot_tree = eval(util.config_val("EXPL", "plot_tree", "False"))
        needs_feats = False
        if plot_feats or tsne or scatter or model_type or plot_tree or shap:
            # these investigations need features to explore
            expr.extract_feats()
            needs_feats = True
            # explore
            expr.init_runmanager()
            expr.runmgr.do_runs()
    expr.analyse_features(needs_feats)
    expr.store_report()
    print("DONE")


if __name__ == "__main__":
    main()
