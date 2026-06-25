"""Tests for nkululeko.bundle and nkululeko.infer modules.

Covers:
- bundle: manifest building, inference.ini generation, feature schema, README
- infer: bundle loading, CLI argument parsing, prediction pipeline
"""

import configparser
import json
import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.constants import VERSION


# ---------------------------------------------------------------------------
# bundle module tests
# ---------------------------------------------------------------------------


class TestBuildManifest:
    """Test _build_manifest helper."""

    def _make_util(self, config_vals=None):
        """Create a mock Util with config_val support."""
        defaults = {
            ("DATA", "target", "emotion"): "emotion",
            ("FEATS", "os.set", "eGeMAPSv02"): "eGeMAPSv02",
            ("FEATS", "scale", False): "standard",
            ("MODEL", "type", "svm"): "svm",
            ("EXP", "language", False): "en",
        }
        if config_vals:
            defaults.update(config_vals)

        util = MagicMock()
        util.config_val.side_effect = lambda s, k, d: defaults.get((s, k, d), d)
        util.exp_is_classification.return_value = True
        return util

    def test_classification_manifest(self):
        from nkululeko.bundle import _build_manifest

        util = self._make_util()
        labels = ["anger", "joy", "neutral"]
        feats_type = ["os"]
        manifest = _build_manifest(
            None, util, labels, feats_type, "standard", "svm"
        )

        assert manifest["bundle_version"] == "1.0"
        assert manifest["nkululeko_version"] == VERSION
        assert manifest["task"] == "classification"
        assert manifest["target"] == "emotion"
        assert manifest["labels"] == ["anger", "joy", "neutral"]
        assert manifest["model_type"] == "svm"
        assert "model" in manifest["artifacts"]
        assert "scaler" in manifest["artifacts"]
        assert "label_encoder" in manifest["artifacts"]
        assert manifest["features"]["type"] == ["os"]
        assert manifest["features"]["scale"] == "standard"

    def test_regression_manifest(self):
        from nkululeko.bundle import _build_manifest

        util = self._make_util()
        util.exp_is_classification.return_value = False
        manifest = _build_manifest(None, util, None, ["wav2vec2"], None, "svr")

        assert manifest["task"] == "regression"
        assert "labels" not in manifest
        assert "scaler" not in manifest["artifacts"]
        assert "label_encoder" not in manifest["artifacts"]

    def test_no_scale(self):
        from nkululeko.bundle import _build_manifest

        util = self._make_util()
        manifest = _build_manifest(None, util, ["a", "b"], ["os"], None, "svm")

        assert "scaler" not in manifest["artifacts"]
        assert "scale" not in manifest["features"]


class TestBuildInferenceIni:
    """Test _build_inference_ini helper."""

    def _make_util(self):
        defaults = {
            ("EXP", "language", False): "en",
            ("FEATS", "os.set", "eGeMAPSv02"): "eGeMAPSv02",
        }
        util = MagicMock()
        util.config_val.side_effect = lambda s, k, d: defaults.get((s, k, d), d)
        return util

    def test_basic_ini(self):
        from nkululeko.bundle import _build_inference_ini

        util = self._make_util()
        config = _build_inference_ini(
            util, ["os"], "standard", "svm", ["anger", "joy"], "emotion"
        )

        assert config["DATA"]["target"] == "emotion"
        assert config["FEATS"]["scale"] == "standard"
        assert config["MODEL"]["type"] == "svm"
        assert "os" in config["FEATS"]["type"]

    def test_no_scale_no_labels(self):
        from nkululeko.bundle import _build_inference_ini

        util = self._make_util()
        config = _build_inference_ini(util, ["wav2vec2"], None, "svr", None, "valence")

        assert config["DATA"]["target"] == "valence"
        assert "scale" not in config["FEATS"]
        assert config["MODEL"]["type"] == "svr"


class TestBuildFeatureSchema:
    """Test _build_feature_schema helper."""

    def test_with_dataframe(self):
        from nkululeko.bundle import _build_feature_schema

        df = pd.DataFrame(np.zeros((5, 3)), columns=["f1", "f2", "f3"])
        schema = _build_feature_schema(df)
        assert schema["columns"] == ["f1", "f2", "f3"]
        assert schema["num_features"] == 3

    def test_none_input(self):
        from nkululeko.bundle import _build_feature_schema

        schema = _build_feature_schema(None)
        assert schema["columns"] == []
        assert schema["num_features"] == 0


class TestBuildReadme:
    """Test _build_readme helper."""

    def test_readme_content(self):
        from nkululeko.bundle import _build_readme

        manifest = {
            "task": "classification",
            "target": "emotion",
            "model_type": "svm",
            "nkululeko_version": "1.7.1",
            "python_version": "3.11.0",
            "labels": ["anger", "joy"],
            "features": {"type": ["os"], "scale": "standard"},
            "artifacts": {"model": "model.pkl", "scaler": "scaler.pkl"},
        }
        readme = _build_readme(manifest, "test_exp")
        assert "# Nkululeko Bundle: test_exp" in readme
        assert "classification" in readme
        assert "emotion" in readme
        assert "python -m nkululeko.infer" in readme


# ---------------------------------------------------------------------------
# infer module tests
# ---------------------------------------------------------------------------


class TestLoadBundle:
    """Test _load_bundle helper."""

    def test_missing_manifest(self, tmp_path):
        from nkululeko.infer import _load_bundle

        with pytest.raises(SystemExit):
            _load_bundle(str(tmp_path))

    def test_valid_bundle(self, tmp_path):
        from nkululeko.infer import _load_bundle

        # Create minimal bundle
        from sklearn.svm import SVC

        model = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0)
        model.fit([[1, 2], [3, 4]], [0, 1])

        manifest = {
            "bundle_version": "1.0",
            "nkululeko_version": VERSION,
            "task": "classification",
            "target": "emotion",
            "model_type": "svm",
            "labels": ["anger", "joy"],
            "features": {"type": ["os"], "scale": "standard"},
            "artifacts": {
                "model": "model.pkl",
                "label_encoder": "label_encoder.pkl",
                "feature_schema": "feature_schema.json",
                "inference_config": "inference.ini",
            },
        }
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with open(tmp_path / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        le.fit(["anger", "joy"])
        with open(tmp_path / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        schema = {"columns": ["f1", "f2"], "num_features": 2}
        with open(tmp_path / "feature_schema.json", "w") as f:
            json.dump(schema, f)

        config = configparser.ConfigParser()
        config["DATA"] = {"target": "emotion"}
        config["FEATS"] = {"type": "['os']", "scale": "standard"}
        config["MODEL"] = {"type": "svm"}
        with open(tmp_path / "inference.ini", "w") as f:
            config.write(f)

        bundle = _load_bundle(str(tmp_path))
        assert bundle["manifest"]["task"] == "classification"
        assert bundle["model_clf"] is not None
        assert bundle["label_encoder"] is not None
        assert bundle["feature_schema"]["num_features"] == 2


class TestInferParser:
    """Test the CLI argument parser for infer module."""

    def test_file_arg(self):
        from nkululeko.infer import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["my_bundle", "--file", "test.wav"])
        assert args.bundle == "my_bundle"
        assert args.file == ["test.wav"]

    def test_folder_arg(self):
        from nkululeko.infer import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["my_bundle", "--folder", "/path/to/audio"])
        assert args.bundle == "my_bundle"
        assert args.folder == "/path/to/audio"

    def test_list_arg(self):
        from nkululeko.infer import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["my_bundle", "--list", "files.csv"])
        assert args.bundle == "my_bundle"
        assert args.list_path == "files.csv"

    def test_outfile_arg(self):
        from nkululeko.infer import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["my_bundle", "--file", "test.wav", "--outfile", "out.csv"]
        )
        assert args.outfile == "out.csv"

    def test_no_input_fails(self):
        from nkululeko.infer import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["my_bundle"])


class TestPredictFromBundle:
    """Test _predict_from_bundle logic."""

    def test_classification_prediction(self, tmp_path):
        from nkululeko.infer import _predict_from_bundle
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Train a tiny model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        clf = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0)
        clf.fit(X_train, y_train)

        scaler = StandardScaler()
        scaler.fit(X_train)

        le = LabelEncoder()
        le.fit(["anger", "joy"])

        bundle = {
            "model_clf": clf,
            "scaler": scaler,
            "label_encoder": le,
            "feature_schema": {"columns": ["f1", "f2"], "num_features": 2},
            "manifest": {"task": "classification"},
        }

        # Create test features
        idx = pd.MultiIndex.from_tuples(
            [(str(tmp_path / "test.wav"), pd.Timedelta(0), pd.Timedelta(seconds=1))],
            names=["file", "start", "end"],
        )
        feats = pd.DataFrame([[2.0, 3.0]], index=idx, columns=["f1", "f2"])

        util = MagicMock()
        results = _predict_from_bundle(feats, bundle, util)

        assert "predicted" in results.columns
        assert results.iloc[0]["predicted"] in ["anger", "joy"]

    def test_regression_prediction(self, tmp_path):
        from nkululeko.infer import _predict_from_bundle
        from sklearn.svm import SVR

        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0.1, 0.5, 0.7, 0.9])
        clf = SVR(C=1.0, kernel="rbf", gamma="scale")
        clf.fit(X_train, y_train)

        bundle = {
            "model_clf": clf,
            "scaler": None,
            "label_encoder": None,
            "feature_schema": {"columns": ["f1", "f2"], "num_features": 2},
            "manifest": {"task": "regression"},
        }

        idx = pd.MultiIndex.from_tuples(
            [(str(tmp_path / "test.wav"), pd.Timedelta(0), pd.Timedelta(seconds=1))],
            names=["file", "start", "end"],
        )
        feats = pd.DataFrame([[2.0, 3.0]], index=idx, columns=["f1", "f2"])

        util = MagicMock()
        results = _predict_from_bundle(feats, bundle, util)

        assert "predicted" in results.columns
        assert isinstance(results.iloc[0]["predicted"], (float, np.floating))


class TestSetupGlobConf:
    """Test _setup_glob_conf initializes global config properly."""

    def test_basic_setup(self):
        from nkululeko.infer import _setup_glob_conf
        import nkululeko.glob_conf as glob_conf

        config = configparser.ConfigParser()
        config["DATA"] = {"target": "emotion"}
        config["FEATS"] = {"type": "['os']"}
        config["MODEL"] = {"type": "svm"}

        bundle = {
            "config": config,
            "label_encoder": None,
            "manifest": {"labels": ["anger", "joy"]},
        }

        _setup_glob_conf(bundle)

        assert glob_conf.config is not None
        assert glob_conf.config["DATA"]["target"] == "emotion"


class TestFindAudioFiles:
    """Test _find_audio_files helper."""

    def test_finds_wav_files(self, tmp_path):
        from nkululeko.infer import _find_audio_files

        (tmp_path / "a.wav").write_text("")
        (tmp_path / "b.mp3").write_text("")
        (tmp_path / "c.txt").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.flac").write_text("")

        files = _find_audio_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert "a.wav" in basenames
        assert "b.mp3" in basenames
        assert "d.flac" in basenames
        assert "c.txt" not in basenames


# ---------------------------------------------------------------------------
# bundle CLI parser test
# ---------------------------------------------------------------------------


class TestBundleParser:
    """Test the CLI argument parser for bundle module."""

    def test_config_required(self):
        from nkululeko.bundle import main

        with pytest.raises(SystemExit):
            with patch("sys.argv", ["bundle"]):
                main()

    def test_config_and_output(self):
        """Test parsing --config and --output args."""
        # Simulate the parser behavior
        from nkululeko.bundle import main

        # We just test that the argparse is set up correctly
        # by checking it doesn't crash with proper args but missing file
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["bundle", "--config", "nonexistent.ini"]):
                main()
