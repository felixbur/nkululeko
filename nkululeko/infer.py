# infer.py
# Inference from a nkululeko bundle directory.

"""Run inference using a nkululeko bundle directory.

A bundle is a self-contained directory exported by ``nkululeko.bundle`` that
contains a trained model, scaler, label encoder, feature schema, and a
sanitized inference config.

Usage:
    python -m nkululeko.infer my_bundle --file audio.wav
    python -m nkululeko.infer my_bundle --folder /path/to/audio_files
    python -m nkululeko.infer my_bundle --list files.csv
"""

import argparse
import ast
import configparser
import json
import os
import pickle
import sys
import tempfile

import audformat
import audiofile
import numpy as np
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


AUDIO_EXTS = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]


def _find_audio_files(folder):
    """Recursively find audio files in a folder."""
    files = []
    for root, _dirs, filenames in os.walk(folder):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower().lstrip(".")
            if ext in AUDIO_EXTS:
                files.append(os.path.join(root, fname))
    return files


def _load_bundle(bundle_dir):
    """Load all bundle artifacts and return them as a dict.

    Returns:
        dict with keys: manifest, config, model, scaler, label_encoder, feature_schema
    """
    bundle_dir = os.path.abspath(bundle_dir)

    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(
            f"ERROR: not a valid bundle directory (missing manifest.json): {bundle_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", {})

    # Load inference config
    ini_path = os.path.join(bundle_dir, artifacts.get("inference_config", "inference.ini"))
    config = configparser.ConfigParser()
    config.read(ini_path)

    # Load model
    model_path = os.path.join(bundle_dir, artifacts.get("model", "model.pkl"))
    if not os.path.isfile(model_path):
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    with open(model_path, "rb") as f:
        model_clf = pickle.load(f)

    # Load scaler (optional)
    scaler = None
    if "scaler" in artifacts:
        scaler_path = os.path.join(bundle_dir, artifacts["scaler"])
        if os.path.isfile(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

    # Load label encoder (optional)
    label_encoder = None
    if "label_encoder" in artifacts:
        le_path = os.path.join(bundle_dir, artifacts["label_encoder"])
        if os.path.isfile(le_path):
            with open(le_path, "rb") as f:
                label_encoder = pickle.load(f)

    # Load feature schema
    feature_schema = None
    if "feature_schema" in artifacts:
        schema_path = os.path.join(bundle_dir, artifacts["feature_schema"])
        if os.path.isfile(schema_path):
            with open(schema_path, "r") as f:
                feature_schema = json.load(f)

    return {
        "manifest": manifest,
        "config": config,
        "model_clf": model_clf,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_schema": feature_schema,
        "bundle_dir": bundle_dir,
    }


def _setup_glob_conf(bundle):
    """Set up glob_conf with the bundle's inference config."""
    config = bundle["config"]
    # Ensure required sections exist
    for section in ("EXP", "DATA", "FEATS", "MODEL"):
        if section not in config:
            config.add_section(section)
    # Set up minimal EXP config for Util to work
    if "root" not in config["EXP"]:
        config["EXP"]["root"] = tempfile.mkdtemp(prefix="nkulu_infer_")
    if "name" not in config["EXP"]:
        config["EXP"]["name"] = "infer"
    if "databases" not in config["DATA"]:
        config["DATA"]["databases"] = "['adhoc']"

    glob_conf.init_config(config)
    glob_conf.set_module("infer")

    if bundle["label_encoder"] is not None:
        glob_conf.set_label_encoder(bundle["label_encoder"])

    labels = bundle["manifest"].get("labels", None)
    if labels is not None:
        glob_conf.set_labels(labels)


def _extract_features(files, bundle, util):
    """Extract features for the given audio files using the bundle config.

    Returns:
        pd.DataFrame of features with audformat segmented index.
    """
    from nkululeko.feature_extractor import FeatureExtractor

    # Build a segmented-index DataFrame for the files
    abs_files = [os.path.abspath(f) for f in files]
    idx = audformat.segmented_index(
        files=abs_files,
        starts=[pd.Timedelta(0)] * len(abs_files),
        ends=[pd.NaT] * len(abs_files),
    )
    idx = audformat.utils.to_segmented_index(idx, allow_nat=False)
    df = pd.DataFrame(index=idx)

    # Get feature types from config
    feats_type = util.config_val_list("FEATS", "type", ["os"])

    # Disable feature caching for inference
    glob_conf.config["FEATS"]["no_reuse"] = "True"
    glob_conf.config["FEATS"]["needs_feature_extraction"] = "True"

    feature_extractor = FeatureExtractor(df, feats_type, "infer", "test")
    feats = feature_extractor.extract()

    return feats


def _predict_from_bundle(feats, bundle, util):
    """Run model prediction on extracted features.

    Returns:
        pd.DataFrame with prediction columns.
    """
    model_clf = bundle["model_clf"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    feature_schema = bundle["feature_schema"]
    manifest = bundle["manifest"]
    is_classification = manifest.get("task") == "classification"

    # Validate feature columns if schema is available
    if feature_schema and feature_schema.get("columns"):
        expected_cols = feature_schema["columns"]
        # Reorder/filter features to match expected schema
        missing = set(expected_cols) - set(feats.columns)
        if missing:
            util.warn(
                f"Features missing {len(missing)} columns expected by bundle schema. "
                "Predictions may be inaccurate."
            )
            # Add missing columns as zeros
            for col in missing:
                feats[col] = 0.0
        feats = feats.reindex(columns=expected_cols, fill_value=0.0)

    # Apply scaling
    if scaler is not None:
        scaled = scaler.transform(feats.values)
        feats = pd.DataFrame(scaled, index=feats.index, columns=feats.columns)

    # Replace NaNs
    feats = feats.fillna(0)

    # Predict
    results = pd.DataFrame(index=feats.index)

    if is_classification and hasattr(model_clf, "predict_proba"):
        probas = model_clf.predict_proba(feats.values)
        classes = model_clf.classes_
        predictions = []

        for i, cls in enumerate(classes):
            if label_encoder is not None:
                label = label_encoder.inverse_transform([int(cls)])[0]
            else:
                label = str(cls)
            results[label] = probas[:, i]

        # Get predicted class
        pred_indices = probas.argmax(axis=1)
        pred_classes = [classes[i] for i in pred_indices]
        if label_encoder is not None:
            pred_labels = label_encoder.inverse_transform(
                [int(c) for c in pred_classes]
            )
        else:
            pred_labels = [str(c) for c in pred_classes]
        results["predicted"] = pred_labels
    else:
        predictions = model_clf.predict(feats.values)
        results["predicted"] = predictions

    return results


def _run_files(files, bundle, util, outfile=None):
    """Run inference on a list of audio files."""
    valid_files = []
    for f in files:
        if not os.path.isfile(f):
            util.warn(f"file not found, skipping: {f}")
            continue
        valid_files.append(f)
    if not valid_files:
        print("ERROR: no valid input files", file=sys.stderr)
        sys.exit(1)

    feats = _extract_features(valid_files, bundle, util)
    results = _predict_from_bundle(feats, bundle, util)

    if outfile:
        results.to_csv(outfile)
        util.debug(f"predictions saved to {outfile}")
    else:
        # Print results per file
        for orig in valid_files:
            absp = os.path.abspath(orig)
            rows = results[results.index.get_level_values("file") == absp]
            if rows.empty:
                print(f"{orig}\tERROR: no prediction produced")
            else:
                for col in rows.columns:
                    val = rows.iloc[0][col]
                    print(f"{orig}\t{col}: {val}")

    return results


def _run_folder(folder, bundle, util, outfile):
    """Run inference on all audio files in a folder."""
    if not os.path.isdir(folder):
        print(f"ERROR: not a folder: {folder}", file=sys.stderr)
        sys.exit(1)

    files = _find_audio_files(folder)
    if not files:
        print(f"ERROR: no audio files found in folder: {folder}", file=sys.stderr)
        sys.exit(1)

    util.debug(f"found {len(files)} audio files in {folder}")
    feats = _extract_features(files, bundle, util)
    results = _predict_from_bundle(feats, bundle, util)

    results.to_csv(outfile)
    util.debug(f"predictions saved to {outfile}")
    print(f"Predictions saved to: {outfile}")
    return results


def _run_list(list_path, bundle, util, outfile):
    """Run inference on files specified in a CSV list."""
    if not os.path.isfile(list_path):
        print(f"ERROR: file not found: {list_path}", file=sys.stderr)
        sys.exit(1)

    # Try reading as audformat first
    try:
        in_df = audformat.utils.read_csv(list_path)
        if audformat.is_segmented_index(in_df.index):
            files = list(set(in_df.index.get_level_values("file")))
        else:
            files = list(in_df.index)
    except (ValueError, AttributeError, KeyError):
        # Fall back to plain text/csv
        files = []
        with open(list_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line.lower() == "file":
                    continue
                if line:
                    files.append(line)

    if not files:
        print(f"ERROR: no files found in list: {list_path}", file=sys.stderr)
        sys.exit(1)

    feats = _extract_features(files, bundle, util)
    results = _predict_from_bundle(feats, bundle, util)

    results.to_csv(outfile)
    util.debug(f"predictions saved to {outfile}")
    print(f"Predictions saved to: {outfile}")
    return results


def infer_from_bundle(bundle_dir, files=None, folder=None, list_path=None, outfile=None):
    """Run inference from a bundle directory.

    This is the programmatic API for bundle inference.

    Args:
        bundle_dir: Path to the bundle directory.
        files: List of audio file paths (mutually exclusive with folder/list_path).
        folder: Folder to scan for audio files.
        list_path: CSV file with audio paths.
        outfile: Output CSV path (used for folder/list modes).

    Returns:
        pd.DataFrame with prediction results.
    """
    bundle = _load_bundle(bundle_dir)
    _setup_glob_conf(bundle)
    util = Util("infer", has_config=True)
    util.debug(f"nkululeko {VERSION}: infer from bundle {bundle_dir}")

    manifest = bundle["manifest"]
    util.debug(
        f"  task={manifest.get('task')}, target={manifest.get('target')}, "
        f"model={manifest.get('model_type')}"
    )

    if files:
        return _run_files(files, bundle, util, outfile)
    elif folder:
        return _run_folder(folder, bundle, util, outfile or "./bundle_predictions.csv")
    elif list_path:
        return _run_list(list_path, bundle, util, outfile or "./bundle_predictions.csv")
    else:
        print(
            "ERROR: provide one of --file, --folder, or --list", file=sys.stderr
        )
        sys.exit(1)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m nkululeko.infer",
        description=(
            "Run inference using a nkululeko bundle directory. "
            "The bundle contains a trained model, scaler, label encoder, "
            "and configuration for self-contained inference."
        ),
    )
    parser.add_argument(
        "bundle",
        help="Path to the bundle directory (created by nkululeko.bundle).",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--file",
        nargs="+",
        metavar="AUDIO",
        help="One or more audio files to predict.",
    )
    src.add_argument(
        "--folder",
        help="Folder to scan recursively for audio files.",
    )
    src.add_argument(
        "--list",
        dest="list_path",
        metavar="CSV",
        help="CSV file listing audio paths.",
    )

    parser.add_argument(
        "--outfile",
        default=None,
        help="Output CSV path for predictions (default: stdout for --file, ./bundle_predictions.csv for --folder/--list).",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    # Accept --file "a.mp3 b.wav" as a single space-separated argument
    if args.file and len(args.file) == 1 and " " in args.file[0].strip():
        args.file = args.file[0].split()

    infer_from_bundle(
        bundle_dir=args.bundle,
        files=args.file,
        folder=args.folder,
        list_path=args.list_path,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
