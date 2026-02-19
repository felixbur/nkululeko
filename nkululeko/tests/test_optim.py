import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from nkululeko.optim import OptimizationRunner


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.__contains__.side_effect = lambda x: x in [
        "OPTIM",
        "MODEL",
        "DATA",
        "EXP",
        "FEATS",
        "REPORT",
    ]
    config.__getitem__.side_effect = lambda x: {
        "OPTIM": {
            "model": "svm",
            "search_strategy": "grid",
            "n_iter": "2",
            "cv_folds": "2",
        },
        "MODEL": {"type": "svm"},
        "DATA": {"target": "label"},
        "EXP": {
            "name": "test_optim",
            "root": "/tmp",
            "runs": "1",
            "epochs": "1",
            "traindevtest": "False",
        },
        "FEATS": {"type": "['os']"},
        "REPORT": {"fresh": "True"},
    }[x]
    config.get.side_effect = lambda section, option, fallback=None: {
        ("MODEL", "tuning_params"): None,
        ("DATA", "target"): "label",
    }.get((section, option), fallback)
    config.add_section = MagicMock()
    config.remove_option = MagicMock()
    config.set = MagicMock()
    return config


@pytest.fixture
def runner(mock_config):
    runner = OptimizationRunner(mock_config)
    runner.util = MagicMock()
    runner.util.high_is_good.return_value = True
    runner.util.exp_is_classification.return_value = True
    runner.util.debug = MagicMock()
    runner.util.error = MagicMock()
    runner.save_results = MagicMock()
    runner.search_strategy = "grid"
    runner.n_iter = 2
    runner.cv_folds = 2
    runner.model_type = "svm"
    return runner


@pytest.fixture
def param_specs():
    return {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]}


def _make_mock_expr():
    """Create a mock experiment with proper train/test data."""
    mock_expr = MagicMock()
    mock_expr.df_train = MagicMock()
    mock_expr.df_train.__getitem__ = MagicMock(return_value=np.array([0, 1, 0, 1]))
    mock_expr.df_train.copy.return_value = mock_expr.df_train
    mock_expr.df_test = MagicMock()
    mock_expr.feats_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    mock_expr.feats_test = np.array([[1, 2], [2, 3]])
    return mock_expr


def _make_mock_search(best_params, best_score, cv_results):
    """Create a mock search CV object."""
    mock_search = MagicMock()
    mock_search.best_params_ = best_params
    mock_search.best_score_ = best_score
    mock_search.cv_results_ = cv_results
    return mock_search


def test_run_sklearn_optimization_grid(runner, param_specs):
    mock_expr = _make_mock_expr()
    mock_search = _make_mock_search(
        best_params={"C": 1.0, "kernel": "linear"},
        best_score=0.9,
        cv_results={
            "params": [
                {"C": 0.1, "kernel": "linear"},
                {"C": 1.0, "kernel": "linear"},
            ],
            "mean_test_score": [0.8, 0.9],
        },
    )

    with patch("nkululeko.experiment.Experiment", return_value=mock_expr), patch(
        "sklearn.model_selection.GridSearchCV", return_value=mock_search
    ), patch("nkululeko.modelrunner.Modelrunner") as mock_Modelrunner, patch(
        "nkululeko.glob_conf.config", runner.config
    ):
        mock_runner_inst = MagicMock()
        mock_runner_inst.model.clf = MagicMock()
        mock_Modelrunner.return_value = mock_runner_inst

        best_params, best_score, all_results = runner._run_sklearn_optimization(
            param_specs
        )

        assert best_params == {"C": 1.0, "kernel": "linear"}
        assert best_score == 0.9
        assert isinstance(all_results, list)
        assert all("params" in r and "score" in r for r in all_results)
        runner.save_results.assert_called_once()


def test_run_sklearn_optimization_random(runner, param_specs):
    runner.search_strategy = "random"
    mock_expr = _make_mock_expr()
    mock_search = _make_mock_search(
        best_params={"C": 0.1, "kernel": "rbf"},
        best_score=0.85,
        cv_results={
            "params": [{"C": 0.1, "kernel": "rbf"}, {"C": 1.0, "kernel": "rbf"}],
            "mean_test_score": [0.85, 0.82],
        },
    )

    with patch("nkululeko.experiment.Experiment", return_value=mock_expr), patch(
        "sklearn.model_selection.RandomizedSearchCV", return_value=mock_search
    ), patch("nkululeko.modelrunner.Modelrunner") as mock_Modelrunner, patch(
        "nkululeko.glob_conf.config", runner.config
    ):
        mock_runner_inst = MagicMock()
        mock_runner_inst.model.clf = MagicMock()
        mock_Modelrunner.return_value = mock_runner_inst

        best_params, best_score, all_results = runner._run_sklearn_optimization(
            param_specs
        )

        assert best_params == {"C": 0.1, "kernel": "rbf"}
        assert best_score == 0.85
        assert isinstance(all_results, list)
        assert all("params" in r and "score" in r for r in all_results)
        runner.save_results.assert_called_once()


def test_parameter_mapping(runner):
    """Test that parameters are correctly mapped for sklearn compatibility."""
    param_specs = {"c_val": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
    sklearn_params = runner._convert_to_sklearn_params(param_specs)

    assert "C" in sklearn_params
    assert "c_val" not in sklearn_params
    assert sklearn_params["C"] == [0.1, 1.0, 10.0]
    assert sklearn_params["kernel"] == ["linear", "rbf"]

    param_specs = {"K_val": [3, 5, 7], "KNN_weights": ["uniform", "distance"]}
    sklearn_params = runner._convert_to_sklearn_params(param_specs)

    assert "n_neighbors" in sklearn_params
    assert "weights" in sklearn_params
    assert "K_val" not in sklearn_params
    assert "KNN_weights" not in sklearn_params
    assert sklearn_params["n_neighbors"] == [3, 5, 7]
    assert sklearn_params["weights"] == ["uniform", "distance"]


def test_run_sklearn_optimization_grid_strategy(runner, param_specs):
    runner.search_strategy = "grid"
    mock_expr = _make_mock_expr()
    mock_search = _make_mock_search(
        best_params={"C": 1.0, "kernel": "linear"},
        best_score=0.9,
        cv_results={
            "params": [
                {"C": 0.1, "kernel": "linear"},
                {"C": 1.0, "kernel": "linear"},
            ],
            "mean_test_score": [0.8, 0.9],
        },
    )

    with patch("nkululeko.experiment.Experiment", return_value=mock_expr), patch(
        "sklearn.model_selection.GridSearchCV", return_value=mock_search
    ) as mock_GridSearchCV, patch(
        "nkululeko.modelrunner.Modelrunner"
    ) as mock_Modelrunner, patch("nkululeko.glob_conf.config", runner.config):
        mock_runner_inst = MagicMock()
        mock_runner_inst.model.clf = MagicMock()
        mock_Modelrunner.return_value = mock_runner_inst

        best_params, best_score, all_results = runner._run_sklearn_optimization(
            param_specs
        )

        assert best_params == {"C": 1.0, "kernel": "linear"}
        assert best_score == 0.9
        assert isinstance(all_results, list)
        assert all("params" in r and "score" in r for r in all_results)
        runner.save_results.assert_called_once()
        mock_GridSearchCV.assert_called_once()


# ============================================================
# Tests for ADM optimization
# ============================================================


@pytest.fixture
def adm_config():
    """Config fixture for ADM optimization tests."""
    config = MagicMock()
    model_section = {
        "type": "adm",
        "loss": "bce",
    }
    optim_section = {
        "model": "adm",
        "search_strategy": "grid",
        "metric": "eer",
        "adm.hidden_dim": "[128, 256]",
        "learning_rate": "[0.0001, 0.001]",
        "batch_size": "[16, 32]",
        "optimizer": "['adamw', 'adam']",
        "feature_noise": "[0.0, 0.01]",
        "threshold": "[0.3, 0.5]",
        "loss": "['bce', 'focal']",
        "focal.alpha": "[0.25, 0.4]",
        "focal.gamma": "[2.0, 3.0]",
    }
    config.__contains__.side_effect = lambda x: x in [
        "OPTIM",
        "MODEL",
        "DATA",
        "EXP",
        "FEATS",
        "REPORT",
    ]
    config.__getitem__.side_effect = lambda x: {
        "OPTIM": optim_section,
        "MODEL": model_section,
        "DATA": {"target": "label"},
        "EXP": {
            "name": "test_adm_optim",
            "root": "/tmp",
            "runs": "1",
            "epochs": "5",
        },
        "FEATS": {"type": "['hubert-base-ls960', 'sptk']"},
        "REPORT": {"fresh": "True"},
    }[x]
    config.get.side_effect = lambda section, option, fallback=None: fallback
    config.add_section = MagicMock()
    config.remove_option = MagicMock()
    config.set = MagicMock()
    return config


@pytest.fixture
def adm_runner(adm_config):
    """ADM optimization runner fixture."""
    runner = OptimizationRunner(adm_config)
    runner.util = MagicMock()
    runner.util.high_is_good.return_value = False  # EER: lower is better
    runner.util.exp_is_classification.return_value = True
    runner.util.debug = MagicMock()
    runner.util.error = MagicMock()
    runner.save_results = MagicMock()
    runner.model_type = "adm"
    runner.metric = "eer"
    return runner


def test_update_adm_params(adm_runner, adm_config):
    """Test that _update_adm_params sets all ADM keys in MODEL config."""
    params = {
        "adm.hidden_dim": 256,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adamw",
        "feature_noise": 0.01,
        "threshold": 0.3,
        "loss": "focal",
        "focal.alpha": 0.25,
        "focal.gamma": 2.0,
    }
    adm_runner._update_adm_params(params)

    model = adm_config["MODEL"]
    assert model["adm.hidden_dim"] == "256"
    assert model["learning_rate"] == "0.001"
    assert model["batch_size"] == "32"
    assert model["optimizer"] == "adamw"
    assert model["feature_noise"] == "0.01"
    assert model["threshold"] == "0.3"
    assert model["loss"] == "focal"
    assert model["focal.alpha"] == "0.25"
    assert model["focal.gamma"] == "2.0"


def test_update_config_routes_adm(adm_runner):
    """Test that _update_config_with_params routes to ADM handler."""
    params = {"adm.hidden_dim": 128, "learning_rate": 0.0001}
    adm_runner._update_config_with_params(params)
    model = adm_runner.config["MODEL"]
    assert model["adm.hidden_dim"] == "128"
    assert model["learning_rate"] == "0.0001"


def test_adm_focal_params_skipped_when_not_focal(adm_runner, adm_config):
    """Test that focal.alpha/gamma are skipped when loss is not focal."""
    params = {
        "loss": "bce",
        "focal.alpha": 0.4,
        "focal.gamma": 3.0,
        "adm.hidden_dim": 256,
    }
    adm_runner._update_adm_params(params)

    model = adm_config["MODEL"]
    assert model["loss"] == "bce"
    assert model["adm.hidden_dim"] == "256"
    # focal params should NOT have been set
    assert "focal.alpha" not in model or model.get("focal.alpha") != "0.4"


def test_adm_focal_params_applied_when_focal(adm_runner, adm_config):
    """Test that focal.alpha/gamma are applied when loss is focal."""
    params = {
        "loss": "focal",
        "focal.alpha": 0.4,
        "focal.gamma": 3.0,
    }
    adm_runner._update_adm_params(params)

    model = adm_config["MODEL"]
    assert model["loss"] == "focal"
    assert model["focal.alpha"] == "0.4"
    assert model["focal.gamma"] == "3.0"


def test_parse_adm_optim_params(adm_runner):
    """Test that OPTIM section with ADM dotted keys is parsed correctly."""
    param_specs = adm_runner.parse_optim_params()

    assert adm_runner.model_type == "adm"
    assert adm_runner.metric == "eer"
    assert "adm.hidden_dim" in param_specs
    assert param_specs["adm.hidden_dim"] == [128, 256]
    assert "learning_rate" in param_specs
    assert param_specs["learning_rate"] == [0.0001, 0.001]
    assert "loss" in param_specs
    assert param_specs["loss"] == ["bce", "focal"]
    assert "focal.alpha" in param_specs
    assert param_specs["focal.alpha"] == [0.25, 0.4]
    assert "focal.gamma" in param_specs
    assert param_specs["focal.gamma"] == [2.0, 3.0]
    assert "optimizer" in param_specs
    assert param_specs["optimizer"] == ["adamw", "adam"]
