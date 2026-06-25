import configparser
import tempfile
import os
import numpy as np
import pickle
import pytest
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

    def analyse_features(self, needs_feats):
        # Test double hook intentionally does nothing.
        pass

    def store_report(self):
        # Test double hook intentionally does nothing.
        pass


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


def test_unsupported_module_exits():
    """run_flags_experiments exits with SystemExit for unknown module names."""
    from nkululeko.flags import run_flags_experiments

    config_file = make_config_file({"models": "['xgb']"})
    with pytest.raises(SystemExit):
        run_flags_experiments(config_file, module="unknown")
    os.remove(config_file)


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


# --- name_target pairs ---

@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_name_target_pairs_count():
    """name_target pairs are expanded as a unit, not combined via product."""
    from nkululeko.flags import run_flags_experiments

    flags = {
        "name_target": "[('exp_a', 'target_a'), ('exp_b', 'target_b')]",
        "models": "['xgb', 'mlp']",
    }
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    # 2 pairs × 2 models = 4 experiments
    assert len(results) == 4


@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_name_target_pairs_only():
    """name_target with no other flags produces one experiment per pair."""
    from nkululeko.flags import run_flags_experiments

    flags = {
        "name_target": "[('grade', 'grade'), ('roughness', 'roughness')]",
    }
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    assert len(results) == 2
    assert all("error" not in r for r in results)


@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_name_target_sets_exp_name_and_target():
    """Each name_target combo sets the correct EXP.name and DATA.target."""
    from nkululeko.flags import run_flags_experiments

    flags = {
        "name_target": "[('myexp', 'mytarget')]",
    }
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file)
    os.remove(config_file)
    assert len(results) == 1
    params = results[0]["parameters"]
    assert params["name_target"] == ("myexp", "mytarget")


# --- explore module ---

@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_explore_module_returns_none_result():
    """Explore module runs without a result score."""
    from nkululeko.flags import run_flags_experiments

    flags = {"models": "['xgb']"}
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file, module="explore")
    os.remove(config_file)
    assert len(results) == 1
    assert results[0]["result"] is None
    assert results[0]["last_epoch"] is None


@patch("nkululeko.experiment.Experiment", DummyExperiment)
def test_explore_module_with_name_target():
    """Explore module works with name_target pairs."""
    from nkululeko.flags import run_flags_experiments

    flags = {
        "name_target": "[('grade', 'grade'), ('roughness', 'roughness')]",
    }
    config_file = make_config_file(flags)
    results = run_flags_experiments(config_file, module="explore")
    os.remove(config_file)
    assert len(results) == 2
    assert all(r["result"] is None for r in results)


# --- _get_importance caching ---

def _make_analyser(tmp_path):
    """Build a FeatureAnalyser with minimal fakes."""
    import pandas as pd
    from nkululeko.feat_extract.feats_analyser import FeatureAnalyser

    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    labels = pd.DataFrame({"emotion": ["a", "b", "a"]})

    with (
        patch("nkululeko.glob_conf.config", {"DATA": {"target": "emotion"}, "EXP": {"root": str(tmp_path), "name": "t"}}),
        patch("nkululeko.utils.util.Util.get_path", return_value=str(tmp_path) + "/"),
        patch("nkululeko.utils.util.Util.config_val", side_effect=lambda s, k, d=None: "emotion" if k == "target" else d),
        patch("nkululeko.plots.Plots.__init__", return_value=None),
    ):
        return FeatureAnalyser("train", labels, features)


def test_get_importance_caches_result(tmp_path):
    """Second call returns cached array without refitting."""
    from sklearn.tree import DecisionTreeClassifier

    analyser = _make_analyser(tmp_path)
    model = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0)

    with patch.object(analyser.util, "get_path", return_value=str(tmp_path) + "/"):
        importance1 = analyser._get_importance(model, permutation=False)
        # Tamper with the cache to confirm second call reads from disk
        cache_path = os.path.join(
            str(tmp_path), "cache", "importance_DecisionTreeClassifier.pkl"
        )
        sentinel = np.array([99.0, 99.0])
        with open(cache_path, "wb") as f:
            pickle.dump(sentinel, f)
        importance2 = analyser._get_importance(model, permutation=False)

    assert importance1 is not None
    np.testing.assert_array_equal(importance2, sentinel)


def test_get_importance_perm_separate_cache(tmp_path):
    """Permutation and non-permutation results are cached independently."""
    from sklearn.tree import DecisionTreeClassifier

    analyser = _make_analyser(tmp_path)
    model = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0)

    with patch.object(analyser.util, "get_path", return_value=str(tmp_path) + "/"):
        analyser._get_importance(model, permutation=False)
        analyser._get_importance(model, permutation=True)

    cache_dir = os.path.join(str(tmp_path), "cache")
    assert os.path.isfile(os.path.join(cache_dir, "importance_DecisionTreeClassifier.pkl"))
    assert os.path.isfile(os.path.join(cache_dir, "importance_DecisionTreeClassifier_perm.pkl"))


def test_get_importance_log_reg_cached(tmp_path):
    """LogisticRegression (coef_-based) importance is cached correctly."""
    from sklearn.linear_model import LogisticRegression

    analyser = _make_analyser(tmp_path)
    model = LogisticRegression(random_state=0)

    with patch.object(analyser.util, "get_path", return_value=str(tmp_path) + "/"):
        importance = analyser._get_importance(model, permutation=False)
        # Tamper with cache to confirm second call reads from disk
        cache_path = os.path.join(str(tmp_path), "cache", "importance_LogisticRegression.pkl")
        sentinel = np.array([42.0, 42.0])
        with open(cache_path, "wb") as f:
            pickle.dump(sentinel, f)
        importance2 = analyser._get_importance(model, permutation=False)

    assert importance is not None
    assert importance.shape == (2,)  # two features
    np.testing.assert_array_equal(importance2, sentinel)
