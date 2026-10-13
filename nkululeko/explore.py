"""Explore the feature sets of a machine learning experiment.

This script is the entry point for the 'explore' module of the nkululeko framework.
It handles loading the experiment configuration, setting up the experiment, and
running various feature exploration techniques based on the configuration.

The script supports the following configuration options:
- `no_warnings`: If set to `True`, it will ignore all warnings during the exploration.
- `feature_distributions`: If set to `True`, it will generate plots of the feature distributions.
- `tsne`: If set to `True`, it will generate a t-SNE plot of the feature space.
- `umap`: If set to `True`, it will generate a UMAP plot of the feature space.
- `pca`: If set to `True`, it will generate a PCA plot of the feature space.
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
import ast
import configparser
import os
from pathlib import Path

import sys

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.dataframe import latest_source_csv_mtime
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.util import Util


def _saved_experiment_is_stale(save_name, config, util):
    """True if a saved experiment pickle at `save_name` is older than a
    CSV-backed dataset's source file listed in DATA.databases (see
    latest_source_csv_mtime) -- i.e. it must not be reused as-is, since a
    column could have been added to the data CSV after it was written.

    False (never stale) whenever there's no mtime signal to compare
    against: no DATA.databases list, no CSV-backed dataset among them
    (e.g. all audb-managed), or no save file to begin with.
    """
    try:
        dataset_names = ast.literal_eval(config["DATA"]["databases"])
    except (KeyError, ValueError, SyntaxError):
        dataset_names = []
    source_mtime = latest_source_csv_mtime(dataset_names, util)
    return (
        source_mtime is not None
        and os.path.isfile(save_name)
        and source_mtime > os.path.getmtime(save_name)
    )


def _should_reuse_saved_experiment(save_name, config, util):
    """True if a saved experiment at `save_name` should be loaded instead
    of starting fresh from load_datasets()/fill_train_and_tests().

    False whenever either DATA.no_reuse is set -- a separate, explicit
    reason to skip it (same convention as should_reuse_split()/
    should_reuse_file(), review on #468), independent of mtime -- or the
    saved experiment is stale per _saved_experiment_is_stale().
    """
    if util.config_val_bool("DATA", "no_reuse", False):
        return False
    return not _saved_experiment_is_stale(save_name, config, util)


def main():
    parser = argparse.ArgumentParser(
        description="Call the nkululeko EXPLORE framework."
    )
    parser.add_argument(
        "config_positional",
        nargs="?",
        default=None,
        metavar="CONFIG",
        help="ini configuration file (positional alternative to --config).",
    )
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    config_file = args.config_positional or args.config

    try:
        if not Path(config_file).is_file():
            print(f"ERROR: no such file: {config_file}")
            sys.exit(1)

        config = configparser.ConfigParser()
        config.read(config_file)
        expr = Experiment(config)
        module = "explore"
        expr.set_module(module)
        util = Util(module)
        util.debug(
            f"running {expr.name} from config {config_file}, nkululeko version {VERSION}"
        )

        if util.config_val("EXP", "no_warnings", False):
            import warnings

            warnings.filterwarnings("ignore")
        # A saved experiment pickle is a second, separate cache layer from
        # the traindf.csv/testdf.csv split cache (should_reuse_split) --
        # loading it below bypasses load_datasets()/Dataset.prepare()
        # entirely, so that cache's own source-file staleness check never
        # runs (review on #468). Apply the same check here: don't reuse a
        # saved experiment that's older than a CSV-backed dataset's source
        # file, or a column added to the data CSV after the experiment was
        # last run stays silently invisible.
        save_name = util.get_save_name()
        if _saved_experiment_is_stale(save_name, config, util):
            util.warn(
                "DATA source file(s) changed since the saved experiment "
                f"{save_name} was written -- loading fresh instead of "
                "reusing it"
            )
        should_reuse = _should_reuse_saved_experiment(save_name, config, util)

        needs_feats = False
        experiment_loaded = False
        if should_reuse:
            try:
                # load the experiment
                expr.load(save_name)
                expr.util.set_config(config)
                needs_feats = True
                experiment_loaded = True
            except FileNotFoundError:
                pass
        if not experiment_loaded:
            # first time (or the saved experiment was ignored as stale):
            # load the data
            expr.load_datasets()

            # split into train and test
            expr.fill_train_and_tests()
            util.debug(
                f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}"
            )

        # Check exploration settings regardless of whether experiment was loaded or not
        plot_feats = eval(util.config_val("EXPL", "feature_distributions", "False"))
        tsne_plot = eval(util.config_val("EXPL", "tsne", "False"))
        umap_plot = eval(util.config_val("EXPL", "umap", "False"))
        pca_plot = eval(util.config_val("EXPL", "pca", "False"))
        scatter = eval(util.config_val("EXPL", "scatter", "False"))
        shap = eval(util.config_val("EXPL", "shap", "False"))
        model_type = util.config_val("EXPL", "model", False)
        plot_tree = eval(util.config_val("EXPL", "plot_tree", "False"))

        if (
            plot_feats
            or tsne_plot
            or umap_plot
            or pca_plot
            or scatter
            or model_type
            or plot_tree
            or shap
        ):
            # these investigations need features to explore
            if not experiment_loaded or not needs_feats:
                expr.extract_feats()
            needs_feats = True
            # explore
            if shap:
                # SHAP analysis requires a trained model
                expr.init_runmanager()
                expr.runmgr.do_runs()
        expr.analyse_features(needs_feats)
        expr.store_report()
        print("DONE")
    except NkululukoError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
