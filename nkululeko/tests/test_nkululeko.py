import pytest
import configparser
import tempfile
import os
from unittest.mock import patch
from pathlib import Path


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
                assert result == 0.85
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
