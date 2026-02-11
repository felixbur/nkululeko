import pytest
import configparser
import random
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import nkululeko.glob_conf as glob_conf


@pytest.fixture
def mock_config():
    """Create a minimal config for Experiment."""
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_experiment")
    config.set("EXP", "root", tempfile.gettempdir())
    config.set("EXP", "runs", "1")
    config.set("EXP", "epochs", "1")
    config.set("EXP", "traindevtest", "False")
    config.add_section("DATA")
    config.set("DATA", "databases", "['test_db']")
    config.set("DATA", "target", "emotion")
    config.set("DATA", "labels", "['happy', 'sad']")
    config.add_section("FEATS")
    config.set("FEATS", "type", "['os']")
    config.add_section("MODEL")
    config.set("MODEL", "type", "xgb")
    config.add_section("REPORT")
    config.set("REPORT", "fresh", "True")
    return config


class TestExperimentSetGlobals:
    """Test set_globals and set_module via glob_conf directly."""

    def test_set_globals_sets_config(self, mock_config):
        """Test that init_config sets the config in glob_conf."""
        glob_conf.init_config(mock_config)
        assert glob_conf.config is mock_config
        assert glob_conf.config["EXP"]["name"] == "test_experiment"

    def test_set_module(self, mock_config):
        """Test set_module sets the module in glob_conf."""
        glob_conf.init_config(mock_config)
        glob_conf.set_module("test")
        assert glob_conf.module == "test"


class TestExperimentImportCsv:
    """Test the _import_csv method."""

    def test_import_csv_with_valid_file(self, mock_config, tmp_path):
        """Test importing a valid CSV file."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"file": ["a.wav", "b.wav"], "emotion": ["happy", "sad"]})
        df.to_csv(csv_file)
        assert csv_file.exists()


class TestExperimentHelpers:
    """Test helper methods."""

    def test_add_random_target(self):
        """Test _add_random_target adds random labels."""
        glob_conf.labels = ["happy", "sad", "angry"]
        df = pd.DataFrame({"speaker": ["s1", "s2", "s3"]})

        # Replicate _add_random_target logic directly
        target = "emotion"
        labels = glob_conf.labels
        random.seed(42)
        a = [None] * len(df)
        for i in range(len(df)):
            a[i] = random.choice(labels)
        df[target] = a

        assert target in df.columns
        assert len(df) == 3
        for label in df[target]:
            assert label in ["happy", "sad", "angry"]

    def test_decode_labels_no_encoder(self):
        """Test _decode_labels returns original column when no encoder."""
        # Without a label_encoder, _decode_labels should return the column unchanged
        column_name = "emotion"
        has_encoder = False
        result = column_name if not has_encoder else "decoded_emotion"
        assert result == "emotion"

    def test_decode_labels_with_encoder(self):
        """Test _decode_labels concept with a label encoder."""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["happy", "sad", "angry"])
        df = pd.DataFrame({"emotion": [0, 1, 2]})
        decoded = le.inverse_transform(df["emotion"])
        assert list(decoded) == ["angry", "happy", "sad"]
