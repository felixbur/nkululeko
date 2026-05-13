"""Unified nkululeko prediction module.

Replaces the previous ``nkululeko.demo``, ``nkululeko.feature_demo`` and
``nkululeko.testing`` modules. Predicts labels from audio in four input modes
and two prediction sources.

Input (mutually exclusive):
    --file FILE [FILE ...]   one or more audio files. Writes per-file
                             ``<name>_result.txt`` next to each input file.
    --folder FOLDER          recursively scan a folder for audio files.
                             Writes a single CSV (see --outfile).
    --list CSV               a CSV containing audio paths. Existing columns
                             and index are preserved. Writes a single CSV.
    --mic                    record from the microphone in a loop and print
                             predictions to stdout.

Prediction source:
    --type feats (default)   use a feature extractor or autopredict target
                             named via --model. Autopredict targets are
                             age, gender, emotion, mos, snr, pesq, sdr, stoi,
                             arousal, valence, dominance, speaker, text,
                             textclassification, translation.
    --type model             load the experiment from --config and use its
                             best trained model.

Examples:
    python -m nkululeko.predict --file test.mp3 test2.wav --model emotion
    python -m nkululeko.predict --list testdata.csv --config config.ini --type model
    python -m nkululeko.predict --mic --config config.ini
"""

import argparse
import ast
import configparser
import os
import sys
import tempfile

import audformat
import audiofile
import numpy as np
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import VERSION, SAMPLING_RATE
from nkululeko.utils.files import find_files
from nkululeko.utils.util import Util


AUTOPREDICT_TARGETS = {
    "speaker",
    "gender",
    "age",
    "snr",
    "mos",
    "pesq",
    "sdr",
    "stoi",
    "text",
    "textclassification",
    "translation",
    "arousal",
    "valence",
    "dominance",
    "emotion",
}

AUDIO_EXTS = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]

DEFAULT_OUTFILE = "./prediction_result.csv"
MIC_DURATION_S = 5
MIC_SAMPLE_RATE = SAMPLING_RATE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m nkululeko.predict",
        description=(
            "Unified nkululeko prediction. Predicts from a feature extractor, "
            "an autopredict target (age, gender, emotion, mos, ...), or a "
            "trained model (--type model)."
        ),
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--file",
        nargs="+",
        metavar="AUDIO",
        help=(
            "One or more audio files. Also accepts a single space-separated "
            "string, e.g. --file 'a.wav b.wav'. Writes per-file "
            "<name>_result.txt."
        ),
    )
    src.add_argument(
        "--folder",
        help="Folder to scan recursively for audio files.",
    )
    src.add_argument(
        "--list",
        dest="list_path",
        metavar="CSV",
        help=(
            "CSV file listing audio paths. Existing columns and index are "
            "preserved in the output."
        ),
    )
    src.add_argument(
        "--mic",
        action="store_true",
        help="Record from microphone and print predictions to stdout (loop).",
    )

    parser.add_argument(
        "--no_playback",
        action="store_true",
        help="Suppress playback of the recording in --mic mode.",
    )
    parser.add_argument(
        "--language",
        help=(
            "Language for the text/translation autopredict targets. "
            "For --model text it sets the Whisper source language "
            "(EXP.language). For --model translation it sets the Google "
            "Translate target language (PREDICT.target_language). "
            "ISO 639-1 code, e.g. 'en', 'de', 'pl'."
        ),
    )
    parser.add_argument(
        "--outfile",
        default=DEFAULT_OUTFILE,
        help=(
            f"Output CSV path for --list/--folder (default: {DEFAULT_OUTFILE})."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Predictor name. Autopredict targets: "
            + ", ".join(sorted(AUTOPREDICT_TARGETS))
            + ". Or a feature-extractor name (wav2vec2, opensmile, audmodel, "
            "emotion2vec, ...)."
        ),
    )
    parser.add_argument(
        "--type",
        dest="ptype",
        default="feats",
        choices=["feats", "model"],
        help=(
            "feats (default): use --model as autopredict target or feature "
            "extractor. model: load best model from experiment defined by "
            "--config."
        ),
    )
    parser.add_argument("--config", help="ini configuration file.")
    return parser


def main():
    args = _build_parser().parse_args()

    # accept --file "a.mp3 b.wav" as a single space-separated argument
    if args.file and len(args.file) == 1 and " " in args.file[0].strip():
        args.file = args.file[0].split()

    module_name = "predict"
    config = _load_config(args)
    if args.language:
        _apply_language_override(config, args.language)
    glob_conf.init_config(config)
    glob_conf.set_module(module_name)
    util = Util(module_name, has_config=True)
    util.debug(f"nkululeko {VERSION}: {module_name}")
    if args.language:
        util.debug(
            f"language override: EXP.language=PREDICT.target_language={args.language}"
        )

    if args.ptype == "model" and not args.config:
        util.error("--type model requires --config CONFIG.ini")

    if args.ptype == "feats" and not args.model:
        feats_type = util.config_val("FEATS", "type", None)
        if feats_type is None:
            util.error(
                "--type feats requires --model or FEATS.type in --config"
            )
        args.model = _first_extractor(feats_type)

    if args.mic:
        _run_mic(args, util)
    elif args.file:
        _run_files(args.file, args, util)
    elif args.folder:
        _run_folder(args.folder, args, util)
    elif args.list_path:
        _run_list(args.list_path, args, util)
    elif args.config:
        # No explicit input — use the dataframe defined by the experiment
        # config. EXP.sample_selection (default "all") picks train / test /
        # all from the loaded dataset(s).
        _run_from_config(args, util)
    else:
        util.error(
            "no input given: provide one of --file, --folder, --list, --mic, "
            "or --config (loads the dataframe defined by the experiment's "
            "[DATA] section, selection via EXP.sample_selection)"
        )

    util.debug("DONE")


# ---------------------------------------------------------------------------
# Backwards-compatible API (for nkululeko.flags)
# ---------------------------------------------------------------------------


def do_test(config_file, outfile):
    """Run the trained model on the test split and store predictions.

    Kept for compatibility with ``nkululeko.flags`` which used to import
    ``do_it`` from the (now removed) ``nkululeko.testing`` module.
    """
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)

    from nkululeko.experiment import Experiment

    expr = Experiment(config)
    expr.set_module("test")
    util = Util("test", has_config=True)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko {VERSION}"
    )

    expr.load(f"{util.get_save_name()}")
    expr.fill_train_and_tests()
    expr.extract_feats()
    result = expr.predict_test_and_save(outfile)
    util.debug("DONE")
    return result, 0


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(args):
    """Load --config or fall back to a minimal in-memory config."""
    if args.config:
        if not os.path.isfile(args.config):
            print(f"ERROR: no such config file: {args.config}", file=sys.stderr)
            sys.exit(1)
        config = configparser.ConfigParser()
        config.read(args.config)
        for section in ("EXP", "DATA", "FEATS", "MODEL"):
            if section not in config:
                config.add_section(section)
        if "root" not in config["EXP"]:
            config["EXP"]["root"] = "./"
        if "name" not in config["EXP"]:
            config["EXP"]["name"] = "predict"
        if "databases" not in config["DATA"]:
            config["DATA"]["databases"] = "['adhoc']"
        return config

    config = configparser.ConfigParser()
    tmp_root = tempfile.mkdtemp(prefix="nkulu_predict_")
    config["EXP"] = {"root": tmp_root, "name": "predict"}
    config["DATA"] = {"databases": "['adhoc']", "target": "predicted"}
    config["FEATS"] = {"no_reuse": "True", "needs_feature_extraction": "True"}
    config["MODEL"] = {}
    return config


def _first_extractor(value):
    """Pick the first extractor name from a FEATS.type config value."""
    if isinstance(value, list):
        return value[0] if value else None
    s = str(value).strip()
    if not s:
        return None
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, list):
            return str(lit[0]) if lit else None
        return str(lit)
    except (ValueError, SyntaxError):
        if "," in s:
            return s.split(",", 1)[0].strip()
        return s


def _apply_language_override(config, language):
    """Write a CLI `--language` override into the config.

    Sets both ``EXP.language`` (consumed by ``--model text`` as the Whisper
    source language) and ``PREDICT.target_language`` (consumed by
    ``--model translation`` as the Google Translate target language). The
    same value is written to both because the two autopredict targets are
    mutually exclusive in a single invocation.
    """
    for section in ("EXP", "PREDICT"):
        if section not in config:
            config.add_section(section)
    config["EXP"]["language"] = language
    config["PREDICT"]["target_language"] = language


def _build_segmented_df(files):
    """Build an empty audformat-segmented DataFrame over the given files.

    NaT ends are resolved to actual file durations: downstream feature
    extractors (via ``audinterface``) emit indices with explicit end times,
    and ``Featureset.filter()`` matches by index equality.
    """
    files = [os.path.abspath(f) for f in files]
    idx = audformat.segmented_index(
        files=files,
        starts=[pd.Timedelta(0)] * len(files),
        ends=[pd.NaT] * len(files),
    )
    return pd.DataFrame(index=_resolve_nat_ends(idx))


def _resolve_nat_ends(idx):
    """Replace NaT ends with actual durations (audinterface will do the same)."""
    return audformat.utils.to_segmented_index(idx, allow_nat=False)


# ---------------------------------------------------------------------------
# Input-mode dispatchers
# ---------------------------------------------------------------------------


def _run_files(files, args, util):
    """Predict for one or more files and write per-file <name>_result.txt."""
    valid_files = []
    for f in files:
        if not os.path.isfile(f):
            util.warn(f"file not found, skipping: {f}")
            continue
        valid_files.append(f)
    if not valid_files:
        util.error("no valid input files")

    seg_df = _build_segmented_df(valid_files)
    preds = _predict_df(seg_df, args, util)

    abs_files = [os.path.abspath(f) for f in valid_files]
    for orig, absp in zip(valid_files, abs_files):
        rows = preds[preds.index.get_level_values("file") == absp]
        out_path = f"{os.path.splitext(orig)[0]}_result.txt"
        with open(out_path, "w") as fh:
            if rows.empty:
                fh.write("ERROR: no prediction produced\n")
                print(f"{orig}\tERROR: no prediction produced")
            else:
                for col in rows.columns:
                    val = rows.iloc[0][col]
                    line = f"{col}: {val}"
                    fh.write(line + "\n")
                    print(f"{orig}\t{line}")
        util.debug(f"wrote {out_path}")


def _run_folder(folder, args, util):
    if not os.path.isdir(folder):
        util.error(f"not a folder: {folder}")
    files = find_files(folder, ext=AUDIO_EXTS)
    if not files:
        util.error(f"no audio files found in folder {folder!r}")
    util.debug(f"found {len(files)} audio files in {folder}")

    seg_df = _build_segmented_df(files)
    preds = _predict_df(seg_df, args, util)
    out_df = seg_df.join(preds, how="left")
    out_df.to_csv(args.outfile)
    util.debug(f"wrote {args.outfile}")


def _run_list(csv_path, args, util):
    """Predict for the audio paths in a CSV, preserving columns and index."""
    if not os.path.isfile(csv_path):
        util.error(f"list file not found: {csv_path}")

    # Try audformat first (handles segmented or filewise indices).
    in_df = None
    is_audformat = False
    try:
        in_df = audformat.utils.read_csv(csv_path)
        is_audformat = True
    except Exception:
        in_df = pd.read_csv(csv_path)

    if is_audformat:
        # For single-column inputs audformat returns an Index/Series rather
        # than a DataFrame. Normalize to an empty-column DataFrame.
        if isinstance(in_df, pd.Index):
            in_df = pd.DataFrame(index=in_df)
        elif isinstance(in_df, pd.Series):
            in_df = in_df.to_frame()

        # in_df has audformat index (segmented or filewise). Ensure segmented.
        if audformat.is_segmented_index(in_df.index):
            seg_index = in_df.index
        else:
            files = list(in_df.index)
            seg_index = audformat.segmented_index(
                files=files,
                starts=[pd.Timedelta(0)] * len(files),
                ends=[pd.NaT] * len(files),
            )
        seg_index = _resolve_nat_ends(seg_index)
        seg_df = in_df.copy()
        seg_df.index = seg_index
    else:
        # plain CSV: first column is the audio path
        if in_df.empty or len(in_df.columns) == 0:
            util.error(f"empty CSV: {csv_path}")
        path_col = in_df.columns[0]
        files = in_df[path_col].astype(str).tolist()
        seg_index = audformat.segmented_index(
            files=[os.path.abspath(f) for f in files],
            starts=[pd.Timedelta(0)] * len(files),
            ends=[pd.NaT] * len(files),
        )
        seg_index = _resolve_nat_ends(seg_index)
        seg_df = in_df.drop(columns=[path_col]).copy()
        seg_df.index = seg_index

    preds = _predict_df(seg_df, args, util)
    # Add prediction columns to seg_df, preserving original columns + index.
    out_df = seg_df.copy()
    for col in preds.columns:
        out_df[col] = preds[col]

    out_path = args.outfile or DEFAULT_OUTFILE
    out_df.to_csv(out_path)
    util.debug(f"wrote {out_path}")


def _run_mic(args, util):
    """Record from the microphone in a loop and print predictions to stdout."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as e:
        util.error(f"microphone support requires sounddevice / soundfile: {e}")

    util.debug(
        f"microphone loop: {MIC_DURATION_S}s @ {MIC_SAMPLE_RATE}Hz. "
        "Press Enter to record, q+Enter to quit."
    )
    while True:
        ans = input("[Enter]=record, q=quit: ").strip().lower()
        if ans == "q":
            break
        print(f"Recording for {MIC_DURATION_S}s...", flush=True)
        recording = sd.rec(
            int(MIC_DURATION_S * MIC_SAMPLE_RATE),
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Recording finished.", flush=True)

        if not getattr(args, "no_playback", False):
            print("Playing back recording...", flush=True)
            try:
                sd.play(recording, MIC_SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                util.warn(f"playback failed: {e}")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, recording, MIC_SAMPLE_RATE)
        try:
            seg_df = _build_segmented_df([tmp.name])
            preds = _predict_df(seg_df, args, util)
            if preds.empty:
                print("(no prediction produced)")
            else:
                row = preds.iloc[0]
                for col, val in row.items():
                    print(f"  {col}: {val}")
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass


def _run_from_config(args, util):
    """Predict over the dataframe defined by the experiment's [DATA] section.

    Triggered when none of --file/--folder/--list/--mic is provided but
    --config is. ``EXP.sample_selection`` (default ``"all"``) selects
    train / test / all from the loaded dataset(s).
    """
    from nkululeko.experiment import Experiment

    expr = Experiment(glob_conf.config)
    expr.set_module("predict")
    util.debug(f"loading dataframe from config {args.config!r}")
    expr.load_datasets()
    expr.fill_train_and_tests()

    sample_selection = util.config_val("EXP", "sample_selection", "all")
    if sample_selection == "all":
        seg_df = pd.concat([expr.df_train, expr.df_test])
    elif sample_selection == "train":
        seg_df = expr.df_train
    elif sample_selection == "test":
        seg_df = expr.df_test
    else:
        util.error(
            f"unknown EXP.sample_selection {sample_selection!r}; "
            "expected one of: all, train, test"
        )
    util.debug(
        f"running over EXP.sample_selection={sample_selection}: "
        f"{len(seg_df)} rows"
    )

    if len(seg_df) == 0:
        util.error("selected dataframe is empty; nothing to predict")

    preds = _predict_df(seg_df, args, util)

    out_df = seg_df.copy()
    for col in preds.columns:
        out_df[col] = preds[col]

    out_path = args.outfile or DEFAULT_OUTFILE
    out_df.to_csv(out_path)
    util.debug(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Prediction backends
# ---------------------------------------------------------------------------


def _predict_df(seg_df, args, util):
    """Dispatch prediction over a segmented-index df. Returns prediction columns."""
    if args.ptype == "model":
        return _predict_with_model(seg_df, args, util)

    model_name = args.model or ""
    if model_name.lower() in AUTOPREDICT_TARGETS:
        return _predict_with_autopredict(seg_df, model_name.lower(), util)

    return _predict_with_features(seg_df, model_name, util)


def _predict_with_autopredict(seg_df, target, util):
    """Use an nkululeko.autopredict.* predictor for one of the known targets."""
    util.debug(f"autopredict target: {target}")
    # ensure DATA.databases is parseable by the predictors
    if "databases" not in glob_conf.config["DATA"]:
        glob_conf.config["DATA"]["databases"] = "['adhoc']"

    df_in = seg_df.copy()
    pred_df = _dispatch_autopredict(target, df_in)
    # return only the newly added prediction column(s) (those not in seg_df)
    new_cols = [c for c in pred_df.columns if c not in seg_df.columns]
    if not new_cols:
        return pd.DataFrame(index=seg_df.index)
    return pred_df[new_cols]


def _dispatch_autopredict(target, df):
    """Construct and run the matching autopredict.* predictor."""
    if target == "speaker":
        from nkululeko.autopredict.ap_sid import SIDPredictor as P
    elif target == "gender":
        from nkululeko.autopredict.ap_gender import GenderPredictor as P
    elif target == "age":
        from nkululeko.autopredict.ap_age import AgePredictor as P
    elif target == "snr":
        from nkululeko.autopredict.ap_snr import SNRPredictor as P
    elif target == "mos":
        from nkululeko.autopredict.ap_mos import MOSPredictor as P
    elif target == "pesq":
        from nkululeko.autopredict.ap_pesq import PESQPredictor as P
    elif target == "sdr":
        from nkululeko.autopredict.ap_sdr import SDRPredictor as P
    elif target == "stoi":
        from nkululeko.autopredict.ap_stoi import STOIPredictor as P
    elif target == "text":
        from nkululeko.autopredict.ap_text import TextPredictor as P
        return P(df, Util("predict")).predict("all")
    elif target == "textclassification":
        from nkululeko.autopredict.ap_textclassifier import (
            TextClassificationPredictor as P,
        )
        return P(df, Util("predict")).predict("all")
    elif target == "translation":
        from nkululeko.autopredict.ap_translate import TextTranslator as P
        return P(df, Util("predict")).predict("all")
    elif target == "arousal":
        from nkululeko.autopredict.ap_arousal import ArousalPredictor as P
    elif target == "valence":
        from nkululeko.autopredict.ap_valence import ValencePredictor as P
    elif target == "dominance":
        from nkululeko.autopredict.ap_dominance import DominancePredictor as P
    elif target == "emotion":
        from nkululeko.autopredict.ap_emotion import EmotionPredictor as P
    else:
        raise ValueError(f"unknown autopredict target: {target}")
    return P(df).predict("all")


def _predict_with_features(seg_df, model_name, util):
    """Extract features for each segment and return them as a DataFrame."""
    if not model_name:
        util.error("--type feats requires --model FEATURE_EXTRACTOR")
    util.debug(f"feature extraction: {model_name}")
    extractor = _get_feature_extractor(model_name, util)

    rows = []
    index_keep = []
    for i, idx in enumerate(seg_df.index.to_list()):
        if isinstance(idx, tuple):
            file, start, end = idx
            offset = (
                start.total_seconds()
                if hasattr(start, "total_seconds")
                else float(start)
            )
            duration = None
            if end is not pd.NaT and end is not None and not pd.isna(end):
                duration = (
                    (end - start).total_seconds()
                    if hasattr(end - start, "total_seconds")
                    else None
                )
                if duration is not None and duration <= 0:
                    duration = None
        else:
            file = idx
            offset = 0
            duration = None

        if not os.path.isfile(file):
            util.warn(f"file not found, skipping: {file}")
            continue

        try:
            signal, sr = audiofile.read(
                file, offset=offset, duration=duration, always_2d=True
            )
            feats = extractor.extract_sample(signal, sr)
            feats = _flatten_features(feats)
            rows.append(feats)
            index_keep.append(idx)
        except Exception as e:
            util.warn(f"failed extracting features for {file}: {e}")
            continue

    if not rows:
        return pd.DataFrame(index=seg_df.index)

    feats_arr = np.array(rows)
    cols = [f"feat_{j}" for j in range(feats_arr.shape[1])]
    out = pd.DataFrame(
        feats_arr,
        index=pd.MultiIndex.from_tuples(index_keep, names=seg_df.index.names),
        columns=cols,
    )
    # reindex to match seg_df.index (missing rows -> NaN)
    return out.reindex(seg_df.index)


def _flatten_features(features):
    if isinstance(features, (int, float)):
        return np.array([features])
    if isinstance(features, tuple):
        return np.array(features)
    if isinstance(features, np.ndarray):
        return features.flatten()
    if isinstance(features, pd.Series):
        return features.values
    if isinstance(features, pd.DataFrame):
        return features.values.flatten()
    return np.array(features).flatten()


def _predict_with_model(seg_df, args, util):
    """Load the experiment from --config and run its best model on each file."""
    from nkululeko.experiment import Experiment

    config = glob_conf.config
    expr = Experiment(config)
    expr.set_module("predict")
    expr.load(f"{util.get_save_name()}")

    model = expr.runmgr.get_best_model()
    lab_enc = getattr(expr, "label_encoder", None)

    # Build a fresh feature extractor from FEATS.type. The pickled
    # extractor inside the experiment is unreliable: experiment.save()
    # strips the inner model/model_interface attributes (to make the
    # object picklable) but leaves `model_loaded=True` behind, so
    # `extract_sample()` would skip the reload and then AttributeError.
    feats_type = util.config_val("FEATS", "type", None)
    if feats_type is None:
        util.error("--type model requires FEATS.type in --config")
    extractor_name = _first_extractor(feats_type)
    feature_extractor = _get_feature_extractor(extractor_name, util)

    is_classification = util.exp_is_classification()
    scale_feats = util.config_val("FEATS", "scale", False)

    pred_rows = []
    index_keep = []
    for idx in seg_df.index.to_list():
        if isinstance(idx, tuple):
            file, start, end = idx
            offset = (
                start.total_seconds()
                if hasattr(start, "total_seconds")
                else float(start)
            )
            duration = None
            if end is not pd.NaT and end is not None and not pd.isna(end):
                duration = (
                    (end - start).total_seconds()
                    if hasattr(end - start, "total_seconds")
                    else None
                )
                if duration is not None and duration <= 0:
                    duration = None
        else:
            file = idx
            offset = 0
            duration = None

        if not os.path.isfile(file):
            util.warn(f"file not found, skipping: {file}")
            continue

        try:
            signal, sr = audiofile.read(
                file, offset=offset, duration=duration, always_2d=True
            )
            features = feature_extractor.extract_sample(signal, sr)
            if scale_feats:
                features = (features - features.mean()) / features.std()
            features = np.nan_to_num(features)
            result_dict = model.predict_sample(features)
        except Exception as e:
            util.warn(f"prediction failed for {file}: {e}")
            continue

        row = {}
        if is_classification:
            if lab_enc is not None:
                for k, v in result_dict.items():
                    try:
                        label = lab_enc.inverse_transform(
                            np.array(int(k)).reshape(1)
                        )[0]
                    except Exception:
                        label = str(k)
                    row[label] = f"{v:.3f}"
                if row:
                    row["predicted"] = max(row, key=lambda c: float(row[c]))
            else:
                for k, v in result_dict.items():
                    row[str(k)] = v
        else:
            row["predicted"] = result_dict

        pred_rows.append(row)
        index_keep.append(idx)

    if not pred_rows:
        return pd.DataFrame(index=seg_df.index)

    out = pd.DataFrame(
        pred_rows,
        index=pd.MultiIndex.from_tuples(index_keep, names=seg_df.index.names),
    )
    return out.reindex(seg_df.index)


# ---------------------------------------------------------------------------
# Feature extractor factory (formerly in feature_demo.py)
# ---------------------------------------------------------------------------


def _get_feature_extractor(model_name, util):
    model_lower = model_name.lower()

    # NB: order matters. Match the more specific aud* extractors first so
    # they're not swallowed by the generic "wav2vec2 in name" substring check
    # below (e.g. "audwav2vec2" contains "wav2vec2").
    if "audwav2vec2" in model_lower:
        from nkululeko.feat_extract.feats_audwav2vec2 import Audwav2vec2Set
        ext = Audwav2vec2Set(model_name, None, model_name)
        ext._load_model()
        return ext
    if "auddim" in model_lower:
        from nkululeko.feat_extract.feats_auddim import AuddimSet
        ext = AuddimSet(model_name, None, model_name)
        ext._load_model()
        return ext

    if "wav2vec2" in model_lower or "wav2vec" in model_lower:
        from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2
        ext = Wav2vec2(model_name, None, model_name)
        ext.init_model()
        return ext
    if "hubert" in model_lower:
        from nkululeko.feat_extract.feats_hubert import Hubert
        ext = Hubert(model_name, None, model_name)
        ext.init_model()
        return ext
    if "wavlm" in model_lower:
        from nkululeko.feat_extract.feats_wavlm import Wavlm
        ext = Wavlm(model_name, None, model_name)
        ext.init_model()
        return ext
    if "whisper" in model_lower:
        from nkululeko.feat_extract.feats_whisper import Whisper
        ext = Whisper(model_name, None, model_name)
        ext.init_model()
        return ext
    if "ast" in model_lower:
        from nkululeko.feat_extract.feats_ast import Ast
        ext = Ast(model_name, None, model_name)
        ext.init_model()
        return ext
    if "emotion2vec" in model_lower:
        from nkululeko.feat_extract.feats_emotion2vec import Emotion2vec
        ext = Emotion2vec(model_name, None, model_name)
        ext.init_model()
        return ext
    if (
        "opensmile" in model_lower
        or "gemaps" in model_lower
        or "compare" in model_lower
    ):
        from nkululeko.feat_extract.feats_opensmile import Opensmileset
        return Opensmileset(model_name, None, model_name)
    if "clap" in model_lower:
        from nkululeko.feat_extract.feats_clap import ClapSet
        ext = ClapSet(model_name, None, model_name)
        ext._load_model()
        return ext
    if "spkrec" in model_lower or "xvect" in model_lower or "ecapa" in model_lower:
        from nkululeko.feat_extract.feats_spkrec import Spkrec
        ext = Spkrec(model_name, None, model_name)
        ext.init_model()
        return ext
    if "trill" in model_lower:
        from nkululeko.feat_extract.feats_trill import TRILLset
        ext = TRILLset(model_name, None, model_name)
        ext._load_model()
        return ext
    if "praat" in model_lower:
        from nkululeko.feat_extract.feats_praat import PraatSet
        return PraatSet(model_name, None, model_name)
    if "audmodel" in model_lower:
        from nkululeko.feat_extract.feats_audmodel import AudmodelSet
        ext = AudmodelSet(model_name, None, model_name)
        ext._load_model()
        return ext
    if "agender" in model_lower:
        from nkululeko.feat_extract.feats_agender_agender import (
            Agender_agenderSet,
        )
        ext = Agender_agenderSet(model_name, None, model_name)
        ext._load_model()
        return ext
    if "squim" in model_lower or "pesq" in model_lower or "sdr" in model_lower:
        from nkululeko.feat_extract.feats_squim import SquimSet
        ext = SquimSet(model_name, None, model_name)
        ext.init_model()
        return ext
    if "mos" in model_lower:
        from nkululeko.feat_extract.feats_mos import MosSet
        ext = MosSet(model_name, None, model_name)
        ext.init_model()
        return ext
    if "snr" in model_lower:
        from nkululeko.feat_extract.feats_snr import SnrSet
        return SnrSet(model_name, None, model_name)
    if "sptk" in model_lower:
        from nkululeko.feat_extract.feats_sptk import SptkSet
        return SptkSet(model_name, None, model_name)

    util.error(f"unknown feature extractor: {model_name}")


if __name__ == "__main__":
    main()
