import pytest
import configparser
import tempfile
import os
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DummyReport:
    class Result:
        test = 0.85

    result = Result()


class DummyRunmanager:
    best_results = [DummyReport()]
    last_epochs = [5]

    def do_runs(self):
        pass

    def get_best_result(self, reports):
        return reports[0]


class DummyExperiment:
    def __init__(self, config):
        self.config = config
        self.name = config.get("EXP", "name", fallback="test")
        self.runmgr = DummyRunmanager()
        self.reports = [DummyReport()]

    def set_module(self, name):
        pass

    def load_datasets(self):
        pass

    def fill_train_and_tests(self):
        pass

    def extract_feats(self):
        pass

    def init_runmanager(self):
        pass

    def run(self):
        return [DummyReport()], [5]

    def get_best_report(self, reports):
        return reports[0]

    def evaluate_per_test_set(self):
        pass

    def store_report(self):
        pass


def make_config_file():
    """Create a minimal config file for testing."""
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_exp")
    config.set("EXP", "root", tempfile.gettempdir())
    config.set("EXP", "runs", "1")
    config.set("EXP", "epochs", "1")
    config.add_section("DATA")
    config.set("DATA", "databases", "['test_db']")
    config.set("DATA", "target", "emotion")
    config.set("DATA", "labels", "['happy', 'sad']")
    config.add_section("FEATS")
    config.set("FEATS", "type", "['os']")
    config.add_section("MODEL")
    config.set("MODEL", "type", "xgb")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ini", mode="w")
    config.write(tmp)
    tmp.close()
    return tmp.name


class TestNkululekoDoit:
    """Test the doit function with mocked Experiment."""

    def test_doit_returns_result_and_epoch(self):
        config_file = make_config_file()
        try:
            with patch("nkululeko.nkululeko.exp.Experiment", DummyExperiment):
                from nkululeko.nkululeko import doit

                result, last_epoch = doit(config_file)
                assert result == pytest.approx(0.85)
                assert isinstance(last_epoch, int)
                assert last_epoch == 5
        finally:
            os.remove(config_file)

    def test_doit_missing_config_exits(self):
        with pytest.raises(SystemExit):
            from nkululeko.nkululeko import doit

            doit("nonexistent_config.ini")

    def test_doit_result_type(self):
        config_file = make_config_file()
        try:
            with patch("nkululeko.nkululeko.exp.Experiment", DummyExperiment):
                from nkululeko.nkululeko import doit

                result, last_epoch = doit(config_file)
                assert isinstance(result, float)
                assert isinstance(last_epoch, int)
        finally:
            os.remove(config_file)


class TestNkululekoFastPath:
    """Fast path: DATA.tests + existing saved experiment → skip training."""

    def test_doit_skips_training_and_returns_test_result(self, tmp_path):
        import nkululeko.glob_conf as nk_glob_conf

        original_config = nk_glob_conf.config
        original_le = nk_glob_conf.label_encoder

        try:
            config = configparser.ConfigParser()
            config.read_dict(
                {
                    "EXP": {
                        "name": "test_exp",
                        "root": str(tmp_path),
                        "runs": "1",
                        "epochs": "1",
                    },
                    "DATA": {
                        "databases": "['train_db']",
                        "tests": "['test_db']",
                        "target": "emotion",
                        "labels": "['happy', 'sad']",
                    },
                    "FEATS": {"type": "['os']"},
                    "MODEL": {"type": "xgb"},
                }
            )
            config_file = tmp_path / "test.ini"
            config_file.write_text("")
            with open(config_file, "w") as fh:
                config.write(fh)

            le = LabelEncoder()
            le.fit(["happy", "sad"])
            calls = []

            class FakeReport:
                class Result:
                    test = 0.77

                result = Result()
                preds = np.array([0, 1])

                def set_id(self, run, epoch):
                    pass

                def print_results(self, epoch, file_name=None):
                    pass

                def plot_confmatrix(self, name, epoch):
                    pass

            class FakeModel:
                run = 0
                epoch = 7

                def reset_test(self, df, feats):
                    pass

                def predict(self):
                    return FakeReport()

            class FakeRunmgr:
                def get_best_model(self):
                    return FakeModel()

            class FakeFastPathExperiment:
                def __init__(self, cfg):
                    self.name = "test_exp"
                    # Initialise glob_conf so Util.config_val/get_save_name work.
                    nk_glob_conf.init_config(cfg)
                    self.label_encoder = le
                    self.target = "emotion"
                    self.labels = ["happy", "sad"]
                    self.df_test = pd.DataFrame(
                        {
                            "emotion": ["happy", "sad"],
                            "class_label": ["happy", "sad"],
                        }
                    )
                    self.feats_test = pd.DataFrame({"f1": [0.1, 0.2]})
                    self.runmgr = FakeRunmgr()

                def set_module(self, name):
                    pass

                def load(self, filename):
                    calls.append("load")

                def fill_tests(self, encode=True):
                    calls.append(f"fill_tests(encode={encode})")

                def extract_test_feats(self):
                    calls.append("extract_test_feats")

                # Training methods must NOT be called in the fast path.
                def load_datasets(self):
                    raise AssertionError("load_datasets called in fast path")

                def fill_train_and_tests(self):
                    raise AssertionError("fill_train_and_tests called in fast path")

                def extract_feats(self):
                    raise AssertionError("extract_feats called in fast path")

                def init_runmanager(self):
                    raise AssertionError("init_runmanager called in fast path")

                def run(self):
                    raise AssertionError("run called in fast path")

                def get_best_report(self, r):
                    return r[0]

                def evaluate_per_test_set(self):
                    pass

                def store_report(self):
                    pass

            with (
                patch("nkululeko.nkululeko.exp.Experiment", FakeFastPathExperiment),
                patch("os.path.isfile", return_value=True),
            ):
                from nkululeko.nkululeko import doit

                result, epoch = doit(str(config_file))

            assert result == pytest.approx(0.77)
            assert epoch == 7
            assert "load" in calls
            assert "fill_tests(encode=False)" in calls
            assert "extract_test_feats" in calls

        finally:
            nk_glob_conf.config = original_config
            nk_glob_conf.label_encoder = original_le


class TestNkululekMain:
    """Test the main function argument parsing."""

    def test_version_string(self):
        from nkululeko.constants import VERSION

        assert isinstance(VERSION, str)
        assert len(VERSION) > 0

    def test_config_file_check(self):
        """Test Path.is_file check for config."""
        assert not Path("nonexistent.ini").is_file()
        config_file = make_config_file()
        assert Path(config_file).is_file()
        os.remove(config_file)
