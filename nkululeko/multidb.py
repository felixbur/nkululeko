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


def _reuse_train(config):
    """True if EXP.reuse_train asks for the row-wise model-reuse mode
    (see main()). Parsed as a truthy string, not eval()'d: config values
    come straight from a user-edited ini file, and eval() both risks
    executing arbitrary code and rejects the common lowercase spelling
    ("true") with a NameError -- matching Util.config_val_bool's
    convention instead, same as _no_reuse() above."""
    if not config.has_section("EXP"):
        return False
    val = config["EXP"].get("reuse_train", "False")
    return str(val).strip().lower() in ("true", "1", "yes")


def _lodo(config):
    """True if EXP.lodo asks for leave-one-dataset-out mode (see main()).
    Same truthy-string parsing as _reuse_train(), for the same reason."""
    if not config.has_section("EXP"):
        return False
    val = config["EXP"].get("lodo", "False")
    return str(val).strip().lower() in ("true", "1", "yes")


def _copy_cached_features(src_dir, dst_dir):
    """Copy whole-database feature cache files (``<db>_<feats_type>_all.*``,
    extracted once per database *before* any train/test split -- see
    Dataset.extract_features()) from src_dir into dst_dir, skipping any
    that already exist there. These are the only feature-cache files safe
    to share across multidb pairs: unlike them, ``feats_train``/
    ``feats_test``/``traindf``/``testdf`` are split-specific and must stay
    scoped to their own pair.

    Matches ``*_all*.*``, not just ``*_all.*``: a featureset with a
    configurable hidden layer (e.g. Wav2vec2Feature.extract(), storage =
    f"{name}_l{layer}.pkl" where name already ends in "_all") writes
    ``<db>_<feats_type>_all_l<layer>.pkl`` instead of ``..._all.pkl``. The
    narrower pattern silently missed those files -- every multidb cell
    using a layered SSL feature re-extracted them from scratch instead of
    reusing the previous cell's cache, with no error to indicate why.
    """
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for cached_file in glob.glob(os.path.join(src_dir, "*_all*.*")):
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
        # EXP.reuse_train: opt-in mode (default off, so existing behavior/
        # results are unchanged for anyone not setting this). The default
        # loop below retrains a fresh model for every (i, j) cell, including
        # the dim-1 cells per row that share the same training data (i is
        # fixed, only the test db j varies) -- dim x more training than the
        # science needs. When on, each row trains exactly once (i == i,
        # with EXP.save forced True) and every other column in that row
        # reuses that saved model via the DATA.tests load-and-eval-only
        # path nkululeko.py already supports (see its `has_tests` branch),
        # instead of retraining. Not combined with CROSSDB.train_extra --
        # the reuse path assumes a single training database per row.
        reuse_train = _reuse_train(config)
        if reuse_train and extra_trains:
            raise NkululukoError(
                "EXP.reuse_train is not supported together with CROSSDB.train_extra"
            )
        if reuse_train and config.has_section("AUGMENT"):
            # aug_train.doit() (dispatched below whenever a cell's config
            # has an [AUGMENT] section) always augments and trains from
            # scratch -- it has no DATA.tests fast path, so an off-diagonal
            # cell here would silently retrain instead of reusing the
            # diagonal model.
            raise NkululukoError(
                "EXP.reuse_train is not supported together with AUGMENT"
            )

        # EXP.lodo: leave-one-dataset-out mode. datasets[i] is held out as
        # the fold's test set; every other entry in `datasets` is pooled for
        # training. This is a different shape of run than the N x N matrix
        # above (one result per fold, not one per (train, test) pair) and
        # can't be built out of that loop's CROSSDB.train_extra pooling --
        # train_extra is read once and shared by every cell in the run, so
        # it's only ever the correct complement for a single fold, not all
        # of them (e.g. datasets=[A,B,C,D,E], train_extra=[B,C,D] gives the
        # right pool for fold "test=E", but fold "test=A" would need
        # [B,C,D,E] instead -- a different, dynamically-computed list per
        # fold, which is exactly what this loop computes).
        lodo = _lodo(config)
        lodo_dev = None
        if lodo:
            lodo_dev = config["EXP"].get("lodo_dev", "").strip() or None
        if lodo and reuse_train:
            raise NkululukoError(
                "EXP.lodo is not supported together with EXP.reuse_train"
            )
        if lodo and extra_trains:
            raise NkululukoError(
                "EXP.lodo is not supported together with CROSSDB.train_extra"
            )
        if lodo and lodo_dev and lodo_dev in datasets:
            raise NkululukoError(
                f"EXP.lodo_dev ({lodo_dev}) must not also appear in "
                "EXP.databases -- it needs to stay out of the fold rotation "
                "to serve as a fixed dev domain in every fold"
            )
        if lodo:
            _run_lodo(
                config, config_file, datasets, lodo_dev, feat_cache_dir, reuse_features
            )
            return

        for i in range(dim):
            # In reuse mode, train the (i, i) diagonal cell first so its
            # saved model exists before any off-diagonal cell in the row
            # tries to load it.
            col_order = (
                [i] + [j for j in range(dim) if j != i] if reuse_train else range(dim)
            )
            for j in col_order:
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
                    if reuse_train:
                        # EXP.save controls the top-level experiment pickle
                        # DATA.tests' fast path checks for; MODEL.save (also
                        # True by default, but a base config could turn it
                        # off) controls whether the actual per-epoch model
                        # weights get written at all. Reuse needs both.
                        config["EXP"]["save"] = "True"
                        if not config.has_section("MODEL"):
                            config.add_section("MODEL")
                        config["MODEL"]["save"] = "True"
                elif reuse_train:
                    # Off-diagonal cell, reuse mode: same DATA.databases/
                    # EXP.name as this row's (i, i) cell, so get_save_name()
                    # / get_path("store") resolve to the model just trained
                    # for dataset i -- DATA.tests triggers nkululeko.py's
                    # load-saved-model-and-evaluate-only path instead of a
                    # fresh training run.
                    train = datasets[i]
                    test = datasets[j]
                    print(f"running train: {train}, test: {test} (reused model)")
                    config["DATA"]["databases"] = f"['{train}']"
                    config["DATA"]["tests"] = f"['{test}']"
                    # Without this, Dataset.split() defaults test's own
                    # split_strategy to speaker_split (~20% of it), not the
                    # whole database -- same "test" value the non-reuse
                    # branch below sets, needed here for the same reason.
                    config["DATA"][f"{test}.split_strategy"] = "test"
                    config["EXP"]["name"] = train
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

                if reuse_train and i != j and np.isnan(results[i, i]):
                    # This row's (i, i) training cell already failed -- an
                    # off-diagonal cell here would find no saved model and
                    # silently fall through to nkululeko.py's normal training
                    # path, which (with DATA.databases holding only the
                    # train db) would train+test on dataset i again while
                    # reporting it as train-i-vs-test-j. Skip explicitly
                    # instead of recording a wrong number.
                    print(
                        f"ERROR: skipping train={datasets[i]}, test={datasets[j]} "
                        f"-- {datasets[i]}'s row-training cell failed, no saved "
                        "model to reuse"
                    )
                    results[i, j] = np.nan
                    last_epochs[i, j] = np.nan
                    continue

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


def _run_lodo(config, config_file, datasets, lodo_dev, feat_cache_dir, reuse_features):
    """Leave-one-dataset-out: for each dataset in `datasets`, train on every
    other dataset in the list (pooled) and test on that one held-out
    dataset. `lodo_dev`, if given, is a fixed dataset excluded from the
    rotation and marked split_strategy=dev in every fold (with
    EXP.traindevtest forced True), so Runmanager gets a genuine held-out dev
    split for early stopping instead of scoring epochs against the fold's
    own test set -- the same fix Phase 1's dev-split harness validated.

    EXP.lodo_runs (default 1) repeats each fold that many times, forcing
    EXP.runs=1 for every repeat. This is deliberate: nkulu() with
    EXP.runs > 1 returns only the single best-of-N-runs result (see
    Experiment.get_best_report -- it picks the best DEV result across runs,
    not their mean), which would fold multiple random seeds into one
    slightly-oracle-ish scalar per fold. Repeating at this level instead and
    averaging the returned per-repeat results is how Phase 1's devsplit
    harness got its honest 5.80% +/- 0.36% -- same approach, generalized to
    every fold here.
    """
    dim = len(datasets)
    lodo_runs = int(config["EXP"].get("lodo_runs", "1"))
    results = np.full((dim, lodo_runs), np.nan)
    last_epochs = np.full((dim, lodo_runs), np.nan)
    for i in range(dim):
        test = datasets[i]
        train_pool = [d for d in datasets if d != test]
        for r in range(lodo_runs):
            fold_config = configparser.ConfigParser()
            fold_config.read(config_file)
            all_dbs = train_pool + [test] + ([lodo_dev] if lodo_dev else [])
            fold_config["DATA"]["databases"] = repr(all_dbs)
            for train_db in train_pool:
                fold_config["DATA"][f"{train_db}.split_strategy"] = "train"
            fold_config["DATA"][f"{test}.split_strategy"] = "test"
            if lodo_dev:
                fold_config["DATA"][f"{lodo_dev}.split_strategy"] = "dev"
                fold_config["EXP"]["traindevtest"] = "True"
            fold_config["EXP"]["runs"] = "1"
            fold_name = f"lodo_{test}" if lodo_runs == 1 else f"lodo_{test}_run{r}"
            fold_config["EXP"]["name"] = fold_name
            print(
                f"running LODO fold {i + 1}/{dim} (run {r + 1}/{lodo_runs}): "
                f"test={test}, train={train_pool}"
                + (f", dev={lodo_dev}" if lodo_dev else "")
            )

            tmp_config = "tmp.ini"
            with open(tmp_config, "w") as tmp_file:
                fold_config.write(tmp_file)
            fold_store = os.path.join(
                os.path.join(fold_config["EXP"]["root"], ""), fold_name, "store"
            )
            if reuse_features:
                _copy_cached_features(feat_cache_dir, fold_store)
            try:
                if fold_config.has_section("AUGMENT"):
                    result, last_epoch = aug_train(tmp_config)
                else:
                    result, last_epoch = nkulu(tmp_config)
            except NkululukoError as e:
                print(
                    f"ERROR: skipping LODO fold test={test} run={r} ({e}); "
                    "leaving it blank"
                )
                continue
            if reuse_features:
                _copy_cached_features(fold_store, feat_cache_dir)
            results[i, r] = float(result)
            last_epochs[i, r] = int(last_epoch)

    print(repr(results))
    if not _all_zero(last_epochs):
        print(repr(last_epochs))
    if _all_failed(results):
        print(
            "ERROR: every LODO fold failed -- see the per-fold errors above. "
            "No results file will be written."
        )
        sys.exit(1)
    if _all_zero(results):
        print(
            "ERROR: all LODO results are exactly 0.0 -- this usually means "
            "every train/test split ended up empty (check DATA config), or a "
            "held-out fold's test set had no labels. No results file will be "
            "written."
        )
        sys.exit(1)
    _write_lodo_results(results, last_epochs, datasets, lodo_dev, config)


def _write_lodo_results(results, last_epochs, datasets, lodo_dev, config):
    """Write results_lodo.txt (per-fold mean +/- std across EXP.lodo_runs
    repeats, plus the overall LODO mean +/- std across fold means) and a bar
    plot with error bars -- LODO produces one result per fold, not a matrix,
    so it needs its own output shape rather than plot_heatmap()'s square
    format.
    """
    metric = _metric_label(config)
    root = os.path.join(config["EXP"]["root"], "")
    os.makedirs(root, exist_ok=True)

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fold_means = np.nanmean(results, axis=1)
        fold_stds = np.nanstd(results, axis=1)
        overall_mean = trunc_to_three(np.nanmean(fold_means))
        overall_std = trunc_to_three(np.nanstd(fold_means))

    lodo_runs = results.shape[1]
    file_name = f"{root}results_lodo.txt"
    with open(file_name, "w") as text_file:
        text_file.write(
            f"LODO mean {metric} across {len(datasets)} folds: "
            f"{overall_mean} (std: {overall_std}, {lodo_runs} run(s)/fold)\n"
        )
        text_file.write(f"folds (test domain): {', '.join(datasets)}\n")
        if lodo_dev:
            text_file.write(f"fixed dev domain: {lodo_dev}\n")
        text_file.write("\n")
        for idx, test_db in enumerate(datasets):
            runs_s = ", ".join(
                "nan" if np.isnan(v) else f"{v:.4f}" for v in results[idx]
            )
            fm = fold_means[idx]
            fs = fold_stds[idx]
            fm_s = "nan" if np.isnan(fm) else f"{fm:.4f}"
            fs_s = "nan" if np.isnan(fs) else f"{fs:.4f}"
            text_file.write(
                f"held out {test_db}: mean {fm_s} std {fs_s}  (runs: {runs_s})\n"
            )

    try:
        fmt = config["PLOT"]["format"]
        plot_name = f"{root}lodo.{fmt}"
    except KeyError:
        plot_name = f"{root}lodo.png"
    plt.figure(figsize=(8, 5))
    x = np.arange(len(datasets))
    plt.bar(
        x,
        np.nan_to_num(fold_means),
        yerr=np.nan_to_num(fold_stds),
        capsize=4,
        color="#4C72B0",
    )
    plt.xticks(x, datasets, rotation=30, ha="right")
    plt.ylabel(metric)
    plt.axhline(
        overall_mean,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"LODO mean {overall_mean}",
    )
    plt.legend()
    plt.title(f"LODO {metric} per held-out fold (mean {overall_mean} ± {overall_std})")
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()


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
