import pytest
import configparser
import tempfile
import os
from unittest.mock import patch, MagicMock


class DummyFeats:
    def __init__(self, shape):
        self.shape = shape

    def copy(self):
        return DummyFeats(self.shape)


class DummyReport:
    class Result:
        test = 0.8

    result = Result()


class DummyExperiment:
    def __init__(self, config):
        self.config = config
        self.feats_train = DummyFeats((10, 5))
        self.feats_test = DummyFeats((5, 5))
        self.df_train = MagicMock()
        self.df_train.copy.return_value = self.df_train
        self.df_test = MagicMock()
        self.df_test.copy.return_value = self.df_test

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

    def copy(self):
        return DummyExperiment(self.config)


def make_config_file(flags_section=None):
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_exp")
    config.set("EXP", "root", ".venv")
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

    config.add_section("FLAGS")
    if flags_section:
        for k, v in flags_section.items():
            config.set("FLAGS", k, str(v))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ini", mode="w")
    config.write(tmp)
    tmp.close()
    return tmp.name


@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_run_flags_experiments_success():
    from nkululeko.flags import run_flags_experiments

    flags = {
        "models": "['xgb', 'mlp']",
        "features": "['os', 'mfcc']",
        "balancing": "['none', 'smote']",
    }
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    # Should run 2*2*2 = 8 experiments
    assert isinstance(results, list)
    assert len(results) == 8
    for res in results:
        assert "parameters" in res
        assert "result" in res
        assert "last_epoch" in res


def test_run_flags_experiments_no_flags_section():
    from nkululeko.flags import run_flags_experiments

    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_exp")
    config.set("EXP", "root", ".venv")
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
    results = run_flags_experiments(tmp.name)
    os.remove(tmp.name)
    assert results == []


@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_run_flags_experiments_flags_not_list():
    from nkululeko.flags import run_flags_experiments

    flags = {"models": "xgb", "features": "os"}
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    # Should run 1*1 = 1 experiment
    assert isinstance(results, list)
    assert len(results) == 1
    assert "parameters" in results[0]
    assert "result" in results[0]


@patch("nkululeko.experiment.Experiment")
def test_run_flags_experiments_feature_extraction_error(MockExperiment):
    from nkululeko.flags import run_flags_experiments

    mock_instance = MagicMock()
    mock_instance.extract_feats.side_effect = Exception("Feature extraction failed")
    MockExperiment.return_value = mock_instance

    flags = {"models": "['xgb']", "features": "['os']"}
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    assert results == []
