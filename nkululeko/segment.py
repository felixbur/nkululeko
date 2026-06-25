"""Segments the samples in the dataset into chunks based on voice activity detection using SILERO VAD [1].

The segmentation results are saved to a file, and the distributions of the original and
segmented durations are plotted.

The module also handles configuration options, such as the segmentation method and sample
selection, and reports the segmentation results.

Usage:
    python3 -m nkululeko.segment [--config CONFIG_FILE] [--file AUDIO_FILE]
                                  [--max_length SECS] [--min_length SECS]
                                  [--output_audio]

Example:
    nkululeko.segment --config tests/exp_androids_segment.ini
    nkululeko.segment --config exp.ini --max_length 30 --output_audio
    nkululeko.segment --file speech.wav --max_length 30 --output_audio

References:
    [1] https://github.com/snakers4/silero-vad
    [2] https://github.com/pyannote/pyannote-audio
"""

import argparse
import configparser
import os
import sys

import audeer
import audiofile
import audformat
import pandas as pd

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.report_item import ReportItem
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.util import Util
from nkululeko.utils.dataframe import segment_silence
from nkululeko.plots import Plots


def _resample_if_needed(signal, sampling_rate, target_sr, resamplers):
    """Resample signal to target_sr. Updates resamplers cache in-place."""
    import torch
    import torchaudio

    if sampling_rate not in resamplers:
        resamplers[sampling_rate] = torchaudio.transforms.Resample(
            sampling_rate, target_sr
        )
    return resamplers[sampling_rate](torch.from_numpy(signal)).numpy(), target_sr


def extract_audio_segments(df_seg, data_dir, util, is_silence=False):
    """Extract audio files for each segment in df_seg.

    Args:
        df_seg: DataFrame with audformat multi-index (file, start, end).
        data_dir: Experiment data directory used to resolve the audio output path.
        util: Util instance for configuration access and logging.
        is_silence: If True, indicates that the segments are silence segments.
    """
    audio_dir = util.config_val("SEGMENT", "audio_dir", "segments")
    if is_silence:
        audio_dir = f"{audio_dir}_silence"
    audio_format = util.config_val("SEGMENT", "audio_format", "wav")
    target_sr = util.config_val("SEGMENT", "sampling_rate", None)
    if target_sr is not None:
        target_sr = int(target_sr)
    # Resolve relative paths against the experiment data directory
    if not os.path.isabs(audio_dir):
        audio_dir = os.path.join(data_dir, audio_dir)
    audeer.mkdir(audio_dir)
    util.debug(f"extracting audio segments to {audio_dir} in format {audio_format}")
    _resamplers = {}  # cache Resample transforms keyed by source SR
    for idx, (file, start, end) in enumerate(df_seg.index):
        start_s = start.total_seconds()
        end_s = end.total_seconds()
        duration = end_s - start_s
        if duration <= 0:
            util.debug(
                f"skipping segment {idx} with non-positive duration:"
                f" {file} [{start_s}-{end_s}]"
            )
            continue
        try:
            signal, sampling_rate = audiofile.read(
                file,
                offset=start_s,
                duration=duration,
                always_2d=True,
            )
        except (OSError, RuntimeError) as e:
            util.debug(f"could not read segment {file} [{start_s}-{end_s}]: {e}")
            continue
        if target_sr is not None and target_sr != sampling_rate:
            signal, sampling_rate = _resample_if_needed(
                signal, sampling_rate, target_sr, _resamplers
            )
        stem = os.path.splitext(os.path.basename(file))[0]
        out_name = f"{stem}_segment_{idx:03d}_{start_s:.1f}-{end_s:.1f}.{audio_format}"
        out_path = os.path.join(audio_dir, out_name)
        try:
            audiofile.write(out_path, signal, sampling_rate)
        except OSError as e:
            util.debug(f"could not write segment {out_path}: {e}")
    util.debug(f"audio segment extraction complete: {audio_dir}")


def _apply_cli_overrides(args, config):
    """Inject CLI argument values into a ConfigParser SEGMENT section."""
    if not config.has_section("SEGMENT"):
        config.add_section("SEGMENT")
    if args.max_length is not None:
        config.set("SEGMENT", "max_length", str(args.max_length))
    if args.min_length is not None:
        config.set("SEGMENT", "min_length", str(args.min_length))
    if args.output_audio:
        config.set("SEGMENT", "output_audio", "True")
    if args.sampling_rate is not None:
        config.set("SEGMENT", "sampling_rate", str(args.sampling_rate))


def _run_file_mode(args):
    """Segment a single audio file without an INI configuration file."""
    if not os.path.isfile(args.file):
        print(f"ERROR: no such file: {args.file}")
        exit()

    # Build a minimal config with only [SEGMENT] so Silero_segmenter can
    # read max_length / min_length via util.config_val without a full INI.
    config = configparser.ConfigParser()
    _apply_cli_overrides(args, config)
    glob_conf.init_config(config)

    util = Util("segment", has_config=True)
    util.debug(f"segmenting file: {args.file}")

    files = pd.Series([args.file])
    df = pd.DataFrame(index=files)
    df.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)

    from nkululeko.segmenting.seg_silero import Silero_segmenter

    segmenter = Silero_segmenter(not_testing=True)
    df_seg = segmenter.segment_dataframe(df)

    df_seg["duration"] = df_seg.index.to_series().map(
        lambda x: (x[2] - x[1]).total_seconds()
    )
    print(df_seg.to_string())

    if args.output_audio:
        data_dir = os.path.dirname(os.path.abspath(args.file))
        extract_audio_segments(df_seg, data_dir, util)

    print("DONE")


def main():
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default=None, help="The base configuration")
    parser.add_argument(
        "--file",
        default=None,
        help="Single audio file to segment (no INI file required)",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=None,
        metavar="SECS",
        help="Maximum segment length in seconds (overrides [SEGMENT] max_length in INI)",
    )
    parser.add_argument(
        "--min_length",
        type=float,
        default=None,
        metavar="SECS",
        help="Minimum segment length in seconds (overrides [SEGMENT] min_length in INI)",
    )
    parser.add_argument(
        "--output_audio",
        action="store_true",
        help="Export audio files for each detected segment (overrides [SEGMENT] output_audio in INI)",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=None,
        metavar="HZ",
        help="Resample exported audio to this rate in Hz (overrides [SEGMENT] sampling_rate in INI);"
        " omit to preserve the original sample rate",
    )
    args = parser.parse_args()

    try:
        if args.file is not None:
            _run_file_mode(args)
            return

        config_file = args.config if args.config is not None else "exp.ini"

        if not os.path.isfile(config_file):
            print(f"ERROR: no such file: {config_file}")
            sys.exit(1)

        config = configparser.ConfigParser()
        config.read(config_file)

        _apply_cli_overrides(args, config)
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

        def calc_dur(x):
            starts = x[1]
            ends = x[2]
            return (ends - starts).total_seconds()

        # segment
        segmented_file = util.config_val("SEGMENT", "result", "segmented")

        method = util.config_val("SEGMENT", "method", "silero")
        sample_selection = util.config_val("EXP", "sample_selection", "all")
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
        result_file = f"{expr.data_dir}/{segmented_file}"
        if result_file.endswith(".csv"):
            result_file = result_file[:-4]
        seg_file_name = f"{result_file}.csv"
        segment_silence_file_name = f"{result_file}_silence.csv"
        if os.path.exists(f"{seg_file_name}") and os.path.exists(
            f"{segment_silence_file_name}"
        ):
            util.debug(
                f"reusing existing result file: {seg_file_name} and {segment_silence_file_name}"
            )
            df_seg = audformat.utils.read_csv(f"{seg_file_name}")
            df_silence = audformat.utils.read_csv(f"{segment_silence_file_name}")
        else:
            util.debug(
                f"segmenting {sample_selection}: {df.shape[0]} samples with {method}"
            )
            if method == "silero":
                from nkululeko.segmenting.seg_silero import Silero_segmenter

                segmenter = Silero_segmenter()
                df_seg = segmenter.segment_dataframe(df)
            elif method == "pyannote":
                from nkululeko.segmenting.seg_pyannote import Pyannote_segmenter

                segmenter = Pyannote_segmenter(config)
                df_seg = segmenter.segment_dataframe(df)
            else:
                util.error(f"unknown segmenter: {method}")
            # segment also the gaps between segments to get a full coverage of the original audio
            with_borders = util.config_val("SEGMENT", "include_silence_borders", "False")
            with_borders = str(with_borders).lower() in ("true", "1", "yes")
            df_silence = segment_silence(df_seg, with_borders=with_borders)

            # plot results
            if "duration" not in df.columns:
                df["duration"] = df.index.to_series().map(lambda x: calc_dur(x))
            df_seg["duration"] = df_seg.index.to_series().map(lambda x: calc_dur(x))
            df_silence["duration"] = df_silence.index.to_series().map(lambda x: calc_dur(x))
            plots = Plots()
            plots.plot_durations(
                df, "original_durations", sample_selection, caption="Original durations"
            )
            plots.plot_durations(
                df_seg,
                "segmented_durations",
                sample_selection,
                caption="Segmented durations",
            )
            plots.plot_durations(
                df_silence,
                "silence_durations",
                sample_selection,
                caption="Silence durations",
            )
            if method == "pyannote":
                util.debug(df_seg[["speaker", "duration"]].groupby(["speaker"]).sum())
                plots.plot_speakers(df_seg, sample_selection)

            # save files
            # remove encoded labels
            df_seg = util.check_class_label(df_seg)
            df_silence = util.check_class_label(df_silence)
            df_seg["duration"] = df_seg.index.to_series().map(lambda x: calc_dur(x))
            df_seg.to_csv(f"{seg_file_name}")
            df_silence["duration"] = df_silence.index.to_series().map(lambda x: calc_dur(x))
            df_silence.to_csv(f"{segment_silence_file_name}")

        if util.config_val("SEGMENT", "output_audio", "False").lower() in (
            "true",
            "1",
            "yes",
        ):
            extract_audio_segments(df_seg, expr.data_dir, util)
            extract_audio_segments(df_silence, expr.data_dir, util, is_silence=True)

        num_before = df.shape[0]
        num_after = df_seg.shape[0]
        util.debug(
            f"saved {seg_file_name} and {segment_silence_file_name} "
            f" to {expr.data_dir}, {num_after} samples (was"
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
    except NkululukoError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()  # use this if you want to state the config file path on command line
