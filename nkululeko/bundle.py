# bundle.py
# Export a trained nkululeko experiment as a self-contained bundle directory
# that can be shared and used for inference without the original training data.

"""Export a trained nkululeko experiment as a portable bundle directory.

The bundle contains everything needed for inference:
    - manifest.json: metadata and artifact paths
    - inference.ini: sanitized config (no training paths or private data)
    - model.pkl: the trained model
    - scaler.pkl: the fitted scaler (if scaling was used)
    - label_encoder.pkl: the label encoder (for classification)
    - feature_schema.json: feature column names and dimensions
    - README.md: human-readable description

Usage:
    python -m nkululeko.bundle --config exp.ini [--output /path/to/bundle]

The bundle can then be used with:
    python -m nkululeko.infer my_bundle --file audio.wav
"""

import argparse
import configparser
import json
import os
import pickle
import platform
import sys

import audeer

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util


def _get_model_path(util, run, epoch):
    """Construct the path to the saved model file."""
    model_dir = util.get_path("model_dir")
    name = f"{util.get_exp_name(only_train=True)}_{run}_{epoch:03d}.model"
    return os.path.join(model_dir, name)


def _get_scaler_obj(expr):
    """Return the fitted scaler object if scaling was used, else None."""
    scaler_feats = getattr(expr, "scaler_feats", None)
    if scaler_feats is None:
        return None
    return getattr(scaler_feats, "scaler", None)


def _build_manifest(expr, util, labels, feats_type, scale_type, model_type):
    """Build the manifest.json content as a dictionary."""
    target = util.config_val("DATA", "target", "emotion")
    is_classification = util.exp_is_classification()

    features_info = {"type": feats_type}
    # Add feature-specific config entries
    for ft in feats_type:
        if ft == "os":
            os_set = util.config_val("FEATS", "os.set", "eGeMAPSv02")
            features_info["os.set"] = os_set
        if ft in ("bert", "hubert", "wav2vec2", "wavlm"):
            model_name = util.config_val("FEATS", f"{ft}.model", "")
            if model_name:
                features_info[f"{ft}.model"] = model_name
    if scale_type:
        features_info["scale"] = scale_type

    artifacts = {"model": "model.pkl", "inference_config": "inference.ini"}
    if scale_type:
        artifacts["scaler"] = "scaler.pkl"
    if is_classification:
        artifacts["label_encoder"] = "label_encoder.pkl"
    artifacts["feature_schema"] = "feature_schema.json"

    manifest = {
        "bundle_version": "1.0",
        "nkululeko_version": VERSION,
        "python_version": platform.python_version(),
        "task": "classification" if is_classification else "regression",
        "target": target,
        "model_type": model_type,
        "artifacts": artifacts,
    }
    if is_classification and labels is not None:
        manifest["labels"] = list(labels)
    manifest["features"] = features_info

    return manifest


def _build_inference_ini(util, feats_type, scale_type, model_type, labels, target):
    """Build a sanitized inference.ini config."""
    config = configparser.ConfigParser()

    # EXP section - minimal
    config["EXP"] = {}
    language = util.config_val("EXP", "language", False)
    if language:
        config["EXP"]["language"] = language

    # DATA section - only target and labels
    config["DATA"] = {"target": target}
    if labels is not None:
        config["DATA"]["labels"] = str(list(labels))

    # FEATS section
    config["FEATS"] = {"type": str(feats_type)}
    for ft in feats_type:
        if ft == "os":
            os_set = util.config_val("FEATS", "os.set", "eGeMAPSv02")
            config["FEATS"]["os.set"] = os_set
        if ft in ("bert", "hubert", "wav2vec2", "wavlm"):
            model_name = util.config_val("FEATS", f"{ft}.model", "")
            if model_name:
                config["FEATS"][f"{ft}.model"] = model_name
    if scale_type:
        config["FEATS"]["scale"] = scale_type

    # MODEL section
    config["MODEL"] = {"type": model_type}

    return config


def _build_feature_schema(feats_train):
    """Build feature schema from training features DataFrame."""
    if feats_train is None:
        return {"columns": [], "num_features": 0}
    columns = list(feats_train.columns)
    return {"columns": columns, "num_features": len(columns)}


def _build_readme(manifest, expr_name):
    """Build a human-readable README.md."""
    task = manifest.get("task", "unknown")
    target = manifest.get("target", "unknown")
    model_type = manifest.get("model_type", "unknown")
    nk_version = manifest.get("nkululeko_version", "unknown")
    labels = manifest.get("labels", [])
    features = manifest.get("features", {})

    lines = [
        f"# Nkululeko Bundle: {expr_name}",
        "",
        f"- **Task:** {task}",
        f"- **Target:** {target}",
        f"- **Model type:** {model_type}",
        f"- **Nkululeko version:** {nk_version}",
        f"- **Python version:** {manifest.get('python_version', 'unknown')}",
        "",
    ]
    if labels:
        lines.append(f"- **Labels:** {', '.join(str(l) for l in labels)}")
        lines.append("")
    if features:
        lines.append(f"- **Feature type(s):** {features.get('type', 'unknown')}")
        if "scale" in features:
            lines.append(f"- **Scaler:** {features['scale']}")
        lines.append("")

    lines.extend(
        [
            "## Usage",
            "",
            "```bash",
            "python -m nkululeko.infer <this_bundle_directory> --file audio.wav",
            "```",
            "",
            "## Contents",
            "",
        ]
    )
    for key, val in manifest.get("artifacts", {}).items():
        lines.append(f"- `{val}`: {key}")
    lines.append("")
    return "\n".join(lines)


def export_bundle(config_file, output_dir=None):
    """Export a trained experiment as a bundle directory.

    Args:
        config_file: Path to the experiment configuration file.
        output_dir: Optional output path. Defaults to <root>/<name>/export.

    Returns:
        Path to the created bundle directory.
    """
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}", file=sys.stderr)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)

    # Create experiment and load saved state
    expr = Experiment(config)
    expr.set_module("bundle")
    util = Util("bundle", has_config=True)
    util.debug(f"nkululeko {VERSION}: bundle export from {config_file}")

    # Load the saved experiment
    save_name = util.get_save_name()
    if not os.path.isfile(save_name):
        util.error(
            f"No saved experiment found at {save_name}. "
            "Run nkululeko first with [EXP] save = True."
        )

    expr.load(save_name)
    glob_conf.set_label_encoder(expr.label_encoder)

    # Gather metadata
    target = util.config_val("DATA", "target", "emotion")
    labels = getattr(expr, "labels", None)
    feats_type = util.config_val_list("FEATS", "type", ["os"])
    scale_type = util.config_val("FEATS", "scale", False)
    if scale_type and scale_type.lower() in ("false", "0", "no", "none", ""):
        scale_type = None
    model_type = util.config_val("MODEL", "type", "svm")

    # Get best model info
    best_report = expr.runmgr.get_best_result(expr.runmgr.best_results)
    run = best_report.run
    epoch = best_report.epoch

    # Determine output directory
    if output_dir is None:
        output_dir = util.config_val(
            "EXPORT",
            "bundle_path",
            os.path.join(expr.root, expr.name, "export"),
        )
    audeer.mkdir(output_dir)
    util.debug(f"exporting bundle to: {output_dir}")

    # 1. Copy model
    model_path = _get_model_path(util, run, epoch)
    if not os.path.isfile(model_path):
        util.error(
            f"Model file not found: {model_path}. "
            "Ensure [MODEL] save = True was set during training."
        )
    dest_model = os.path.join(output_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model_data = f.read()
    with open(dest_model, "wb") as f:
        f.write(model_data)
    util.debug(f"  model -> {dest_model}")

    # 2. Save scaler
    if scale_type:
        scaler_obj = _get_scaler_obj(expr)
        if scaler_obj is not None:
            dest_scaler = os.path.join(output_dir, "scaler.pkl")
            with open(dest_scaler, "wb") as f:
                pickle.dump(scaler_obj, f)
            util.debug(f"  scaler -> {dest_scaler}")
        else:
            # No scaler object available, skip
            scale_type = None

    # 3. Save label encoder
    label_encoder = getattr(expr, "label_encoder", None)
    if label_encoder is not None:
        dest_le = os.path.join(output_dir, "label_encoder.pkl")
        with open(dest_le, "wb") as f:
            pickle.dump(label_encoder, f)
        util.debug(f"  label_encoder -> {dest_le}")

    # 4. Feature schema
    feats_train = getattr(expr, "feats_train", None)
    schema = _build_feature_schema(feats_train)
    dest_schema = os.path.join(output_dir, "feature_schema.json")
    with open(dest_schema, "w") as f:
        json.dump(schema, f, indent=2)
    util.debug(f"  feature_schema -> {dest_schema}")

    # 5. Manifest
    manifest = _build_manifest(expr, util, labels, feats_type, scale_type, model_type)
    dest_manifest = os.path.join(output_dir, "manifest.json")
    with open(dest_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    util.debug(f"  manifest -> {dest_manifest}")

    # 6. Inference config
    inference_config = _build_inference_ini(
        util, feats_type, scale_type, model_type, labels, target
    )
    dest_ini = os.path.join(output_dir, "inference.ini")
    with open(dest_ini, "w") as f:
        inference_config.write(f)
    util.debug(f"  inference.ini -> {dest_ini}")

    # 7. README
    expr_name = getattr(expr, "name", "nkululeko_model")
    readme = _build_readme(manifest, expr_name)
    dest_readme = os.path.join(output_dir, "README.md")
    with open(dest_readme, "w") as f:
        f.write(readme)
    util.debug(f"  README.md -> {dest_readme}")

    util.debug(f"bundle export complete: {output_dir}")
    print(f"Bundle exported to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        prog="python -m nkululeko.bundle",
        description=(
            "Export a trained nkululeko experiment as a portable bundle "
            "directory for sharing and inference."
        ),
    )
    parser.add_argument(
        "--config", required=True, help="The experiment configuration file."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the bundle. Defaults to <root>/<name>/export.",
    )
    args = parser.parse_args()
    export_bundle(args.config, args.output)


if __name__ == "__main__":
    main()
