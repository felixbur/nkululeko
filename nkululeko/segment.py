# segment.py
# segment data splits

import argparse
import os
import pandas as pd
import configparser
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util
from nkululeko.constants import VERSION
import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.report_item import ReportItem


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

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    # create a new experiment
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
        from datetime import datetime

        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()

    if "duration" not in df.columns:
        df["duration"] = df.index.to_series().map(lambda x: calc_dur(x))
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
    target = util.config_val("DATA", "target", "emotion")
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


def get_segmentation(file):
    #    print(f'segmenting {file[0]}')
    print(".", end="")
    wav = read_audio(file[0], sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=SAMPLING_RATE
    )
    files, starts, ends = [], [], []
    for entry in speech_timestamps:
        start = float(entry["start"] / 1000.0)
        end = float(entry["end"] / 1000.0)
        files.append(file[0])
        starts.append(start)
        ends.append(end)
    seg_index = segmented_index(files, starts, ends)
    return seg_index


def segment_dataframe(df):
    dfs = []
    for file, values in df.iterrows():
        index = get_segmentation(file)
        dfs.append(
            pd.DataFrame(
                values.to_dict(),
                index,
            )
        )
    return audformat.utils.concat(dfs)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
