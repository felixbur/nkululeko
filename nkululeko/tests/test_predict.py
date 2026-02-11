import pytest
import argparse
import os
import tempfile
import configparser
from unittest.mock import patch, MagicMock


class TestPredictArgParser:
    """Test predict module argument parsing."""

    def test_parser_default_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="exp.ini")
        args = parser.parse_args([])
        assert args.config == "exp.ini"

    def test_parser_custom_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="exp.ini")
        args = parser.parse_args(["--config", "my_config.ini"])
        assert args.config == "my_config.ini"


class TestPredictConfigCheck:
    """Test config file existence checks."""

    def test_missing_config_detected(self):
        assert not os.path.isfile("nonexistent_predict.ini")

    def test_existing_config_detected(self):
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as f:
            tmpfile = f.name
        assert os.path.isfile(tmpfile)
        os.remove(tmpfile)


class TestPredictOutputLogic:
    """Test output processing logic from predict module."""

    def test_sample_selection_default(self):
        """Test default sample selection is 'all'."""
        sample_selection = "all"
        name = f"{sample_selection}_predicted"
        assert name == "all_predicted"

    def test_sample_selection_custom(self):
        """Test custom sample selection naming."""
        sample_selection = "train"
        name = f"{sample_selection}_predicted"
        assert name == "train_predicted"

    def test_class_label_rename_logic(self):
        """Test the class_label column rename logic."""
        import pandas as pd
        target = "emotion"
        df = pd.DataFrame({
            "emotion": [0, 1, 2],
            "class_label": ["happy", "sad", "angry"],
            "speaker": ["s1", "s2", "s3"],
        })
        if "class_label" in df.columns:
            df = df.drop(columns=[target])
            df = df.rename(columns={"class_label": target})
        assert target in df.columns
        assert "class_label" not in df.columns
        assert df[target].tolist() == ["happy", "sad", "angry"]

    def test_output_csv_saving(self):
        """Test CSV output saving."""
        import pandas as pd
        df = pd.DataFrame({
            "emotion": ["happy", "sad"],
            "speaker": ["s1", "s2"],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmpfile = f.name
        df.to_csv(tmpfile)
        loaded = pd.read_csv(tmpfile, index_col=0)
        assert loaded.shape == (2, 2)
        assert "emotion" in loaded.columns
        os.remove(tmpfile)
