import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from nkululeko.optim import OptimizationRunner


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.__contains__.side_effect = lambda x: x in [
        "OPTIM", "MODEL", "DATA", "EXP", "FEATS", "REPORT",
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
    mock_expr.df_train.__getitem__ = MagicMock(
        return_value=np.array([0, 1, 0, 1])
    )
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

    with (
        patch("nkululeko.experiment.Experiment", return_value=mock_expr),
        patch("sklearn.model_selection.GridSearchCV", return_value=mock_search),
        patch("nkululeko.modelrunner.Modelrunner") as mock_Modelrunner,
        patch("nkululeko.glob_conf.config", runner.config),
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

    with (
        patch("nkululeko.experiment.Experiment", return_value=mock_expr),
        patch(
            "sklearn.model_selection.RandomizedSearchCV", return_value=mock_search
        ),
        patch("nkululeko.modelrunner.Modelrunner") as mock_Modelrunner,
        patch("nkululeko.glob_conf.config", runner.config),
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

    with (
        patch("nkululeko.experiment.Experiment", return_value=mock_expr),
        patch(
            "sklearn.model_selection.GridSearchCV", return_value=mock_search
        ) as mock_GridSearchCV,
        patch("nkululeko.modelrunner.Modelrunner") as mock_Modelrunner,
        patch("nkululeko.glob_conf.config", runner.config),
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
        mock_GridSearchCV.assert_called_once()
