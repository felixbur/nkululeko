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
import glob
import os
import shutil
import sys
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from nkululeko.aug_train import doit as aug_train
from nkululeko.nkululeko import doit as nkulu
from nkululeko.utils.errors import NkululukoError


def _no_reuse(config):
    """True if DATA.no_reuse or FEATS.no_reuse asks for fresh extraction,
    matching Util.config_val_bool's truthy-string convention."""
    for section in ("DATA", "FEATS"):
        if config.has_section(section):
            val = config[section].get("no_reuse", "False")
            if str(val).strip().lower() in ("true", "1", "yes"):
                return True
    return False


def _copy_cached_features(src_dir, dst_dir):
    """Copy whole-database feature cache files (``<db>_<feats_type>_all.*``,
    extracted once per database *before* any train/test split -- see
    Dataset.extract_features()) from src_dir into dst_dir, skipping any
    that already exist there. These are the only feature-cache files safe
    to share across multidb pairs: unlike them, ``feats_train``/
    ``feats_test``/``traindf``/``testdf`` are split-specific and must stay
    scoped to their own pair.
    """
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for cached_file in glob.glob(os.path.join(src_dir, "*_all.*")):
        dest = os.path.join(dst_dir, os.path.basename(cached_file))
        if not os.path.isfile(dest):
            shutil.copy2(cached_file, dest)


def main():
    parser = argparse.ArgumentParser(
        description="Call the nkululeko MULTIDB framework."
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
        # Every pair gets its own <root>/<pair_name>/store/ dir (see
        # Util.get_path), so the same database's whole-database features
        # would otherwise be re-extracted for every pair it appears in.
        # Share just those (safe, split-independent) files across pairs
        # via one fixed cache dir, unless the user explicitly wants fresh
        # extraction every time.
        exp_root = os.path.join(config["EXP"]["root"], "")
        feat_cache_dir = os.path.join(exp_root, "_feat_cache")
        reuse_features = not _no_reuse(config)
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
                pair_store = os.path.join(exp_root, config["EXP"]["name"], "store")
                if reuse_features:
                    _copy_cached_features(feat_cache_dir, pair_store)
                try:
                    if config.has_section("AUGMENT"):
                        result, last_epoch = aug_train(tmp_config)
                    else:
                        result, last_epoch = nkulu(tmp_config)
                except NkululukoError as e:
                    # A single pair's data/config problem (e.g. a split
                    # leaving train/test label sets that don't overlap)
                    # shouldn't abort the whole run -- skip this cell and
                    # keep going so every other pair still gets reported.
                    print(f"ERROR: skipping this pair ({e}); leaving it blank")
                    results[i, j] = np.nan
                    last_epochs[i, j] = np.nan
                    continue
                if reuse_features:
                    _copy_cached_features(pair_store, feat_cache_dir)
                results[i, j] = float(result)
                last_epochs[i, j] = int(last_epoch)
        print(repr(results))
        if not _all_zero(last_epochs):
            print(repr(last_epochs))
        if _all_failed(results):
            print(
                "ERROR: every multidb pair failed -- see the per-pair errors "
                "above. No heatmap will be written."
            )
            sys.exit(1)
        if _all_zero(results):
            print(
                "ERROR: all multidb results are exactly 0.0 -- this usually means "
                "every train/test split ended up empty (check DATA/CROSSDB config), "
                "or a saved model was reused against an unlabeled test set "
                "(DATA.tests). No heatmap will be written."
            )
            sys.exit(1)
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
    if np.isnan(x):
        return float("nan")
    return int(x * 1000) / 1000.0


def _all_zero(results):
    """True if no cell in the results matrix is a real, nonzero result.

    A silent 0.0 (e.g. from an empty train/test split, or a reused saved
    model evaluated without labels -- see Reporter's default Result(0, 0, 0,
    0, "unknown") and nkululeko.py's DATA.tests reuse branch) is a valid
    single result, but a matrix that's *entirely* 0.0 (or NaN, from a
    per-pair failure -- see _all_failed) means the whole multidb run
    produced nothing meaningful and shouldn't be plotted.
    """
    return not np.any(np.nan_to_num(results))


def _all_failed(results):
    """True if every single cell failed (e.g. a data/config error caught
    in main()'s per-pair try/except, recorded as NaN) -- nothing was
    computed at all, as opposed to _all_zero's "computed but 0.0"."""
    return bool(np.all(np.isnan(results)))


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
    dim = results.shape[0]
    # Diagonal-excluded ("cross") means per row/column, for the heatmap's
    # extra Mean row/column (issue #424): a train-db's row mean is its
    # average performance on *other* test-dbs, and vice versa for a
    # test-db's column mean.
    masked = results.astype(float).copy()
    np.fill_diagonal(masked, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        # dim == 1 leaves masked entirely NaN (no cross-db result exists);
        # a failed pair (main()'s per-pair try/except) also leaves NaN
        # cells. Either way nanmean may warn "Mean of empty slice" and
        # return nan, which is the correct value here, not an error.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_means = np.nanmean(masked, axis=1)
        col_means = np.nanmean(masked, axis=0)
        mean_non_diag_raw = np.nanmean(masked)

        df_cm = pd.DataFrame(
            results, index=[i for i in labels], columns=[i for i in labels]
        )
        df_cm["mean (cross)"] = row_means
        df_cm.loc["mean (cross)"] = np.append(col_means, mean_non_diag_raw)

        # nanmean throughout: a single failed pair (NaN) must not turn
        # every other, still-valid aggregate into NaN as well.
        mean = trunc_to_three(np.nanmean(results))
        mean_diag = trunc_to_three(np.nanmean(results.diagonal()))
        mean_non_diag = trunc_to_three(mean_non_diag_raw)
        colsums = np.nanmean(results, axis=0)
        vfunc = np.vectorize(trunc_to_three)
        colsums = vfunc(colsums)
        rowsums = np.nanmean(results, axis=1)
        rowsums = vfunc(rowsums)
        colsums_epochs = np.nanmean(last_epochs, axis=0)
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
        # A non-ANN model forces a single epoch, so last_epochs is always
        # exactly 0 everywhere -- not useful information, so skip it.
        if not _all_zero(last_epochs):
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
    main()
