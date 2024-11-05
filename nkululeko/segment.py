"""
Segments the samples in the dataset into chunks based on voice activity detection using SILERO VAD [1].

The segmentation results are saved to a file, and the distributions of the original and
segmented durations are plotted.

The module also handles configuration options, such as the segmentation method and sample
selection, and reports the segmentation results.

Usage:
    python3 -m nkululeko.segment [--config CONFIG_FILE]

Example:
    nkululeko.segment --config tests/exp_androids_segment.ini

References:
    [1] https://github.com/snakers4/silero-vad
    
"""

import argparse
import configparser
import os

import pandas as pd

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.report_item import ReportItem
from nkululeko.utils.util import Util


def main():
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    config_file = args.config if args.config is not None else "exp.ini"

    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    config = configparser.ConfigParser()
    config.read(config_file)
    expr = Experiment(config)
    module = "segment"
    expr.set_module(module)
    util = Util(module)
    util.debug(f"running {expr.name}, nkululeko version {VERSION}")

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    # segment
    segmented_file = util.config_val("SEGMENT", "result", "segmented.csv")

    segmenter = util.config_val("SEGMENT", "method", "silero")
    sample_selection = util.config_val("SEGMENT", "sample_selection", "all")
    if sample_selection == "all":
        df = pd.concat([expr.df_train, expr.df_test])
    elif sample_selection == "train":
        df = expr.df_train
    elif sample_selection == "test":
        df = expr.df_test
    else:
        util.error(
            f"unknown segmentation selection specifier {sample_selection},"
            " should be [all | train | test]"
        )
    util.debug(f"segmenting {sample_selection}: {df.shape[0]} samples with {segmenter}")
    if segmenter == "silero":
        from nkululeko.segmenting.seg_silero import Silero_segmenter

        segmenter = Silero_segmenter()
        df_seg = segmenter.segment_dataframe(df)

    else:
        util.error(f"unkown segmenter: {segmenter}")

    def calc_dur(x):

        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()

    if "duration" not in df.columns:
        df["duration"] = df.index.to_series().map(lambda x: calc_dur(x))
    df_seg["duration"] = df_seg.index.to_series().map(lambda x: calc_dur(x))
    num_before = df.shape[0]
    num_after = df_seg.shape[0]
    # plot distributions
    from nkululeko.plots import Plots

    plots = Plots()
    plots.plot_durations(
        df, "original_durations", sample_selection, caption="Original durations"
    )
    plots.plot_durations(
        df_seg, "segmented_durations", sample_selection, caption="Segmented durations"
    )
    print("")
    # remove encoded labels
    target = util.config_val("DATA", "target", None)
    if "class_label" in df_seg.columns:
        df_seg = df_seg.drop(columns=[target])
        df_seg = df_seg.rename(columns={"class_label": target})
    # save file
    # dataname = "_".join(expr.datasets.keys())
    # name = f"{dataname}{segment_target}"
    df_seg.to_csv(f"{expr.data_dir}/{segmented_file}")
    util.debug(
        f"saved {segmented_file} to {expr.data_dir}, {num_after} samples (was"
        f" {num_before})"
    )
    glob_conf.report.add_item(
        ReportItem(
            "Data",
            "Segmentation",
            f"Segmented {num_before} samples to {num_after} segments",
        )
    )
    expr.store_report()
    print("DONE")


if __name__ == "__main__":
    main()  # use this if you want to state the config file path on command line
