import pytest
import configparser
import tempfile
import os
import sys
import types


class DummyExperiment:
    def __init__(self, config):
        self.config = config
        self.feats_train = type("DummyFeats", (), {"shape": (10, 5)})()
        self.feats_test = type("DummyFeats", (), {"shape": (5, 5)})()
        self.df_train = "dummy_train"
        self.df_test = "dummy_test"

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
        # Return dummy reports and epochs
        class DummyReport:
            class Result:
                test = 0.8

            result = Result()

        return [DummyReport()], [5]

    def get_best_report(self, reports):
        return reports[0]

    def copy(self):
        return DummyExperiment(self.config)


# Patch nkululeko.experiment.Experiment for testing before importing flags
dummy_exp_mod = types.ModuleType("nkululeko.experiment")
setattr(dummy_exp_mod, "Experiment", DummyExperiment)
sys.modules["nkululeko.experiment"] = dummy_exp_mod

# Now import after patching
from nkululeko.flags import run_flags_experiments


def make_config_file(flags_section=None):
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_exp")
    config.set("EXP", "root", ".venv")  # Add required root parameter
    config.set("EXP", "runs", "1")
    config.set("EXP", "epochs", "1")

    # Add required DATA section
    config.add_section("DATA")
    config.set("DATA", "databases", "['test_db']")
    config.set("DATA", "target", "emotion")
    config.set("DATA", "labels", "['happy', 'sad']")

    # Add required FEATS section
    config.add_section("FEATS")
    config.set("FEATS", "type", "['os']")

    # Add required MODEL section
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


def test_run_flags_experiments_success():
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
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_exp")
    config.set("EXP", "root", ".venv")  # Add required root parameter
    config.set("EXP", "runs", "1")
    config.set("EXP", "epochs", "1")

    # Add required DATA section
    config.add_section("DATA")
    config.set("DATA", "databases", "['test_db']")
    config.set("DATA", "target", "emotion")
    config.set("DATA", "labels", "['happy', 'sad']")

    # Add required FEATS section
    config.add_section("FEATS")
    config.set("FEATS", "type", "['os']")

    # Add required MODEL section
    config.add_section("MODEL")
    config.set("MODEL", "type", "xgb")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ini", mode="w")
    config.write(tmp)
    tmp.close()
    results = run_flags_experiments(tmp.name)
    os.remove(tmp.name)
    assert results == []


def test_run_flags_experiments_flags_not_list():
    flags = {"models": "xgb", "features": "os"}  # Use plain strings without quotes
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    # Should run 1*1 = 1 experiment
    assert isinstance(results, list)
    assert len(results) == 1
    assert "parameters" in results[0]
    assert "result" in results[0]


def test_run_flags_experiments_feature_extraction_error(monkeypatch):
    # Patch DummyExperiment.extract_feats to raise Exception
    def raise_error(self):
        raise Exception("Feature extraction failed")

    monkeypatch.setattr(DummyExperiment, "extract_feats", raise_error)
    flags = {"models": "['xgb']", "features": "['os']"}
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    assert results == []
