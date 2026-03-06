"""Unit tests for Reporter class (nkululeko/reporting/reporter.py)."""

import configparser

import numpy as np
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.reporter import Reporter
from nkululeko.reporting.result import Result


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


# Small, deterministic data for classification (3 classes, 30 samples)
TRUTHS_CLS = np.array([0, 1, 2] * 10)
PREDS_CLS = np.array([0, 1, 2] * 10)  # perfect predictions

# Regression data
TRUTHS_REG = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4, dtype=float)
PREDS_REG = np.array([1.1, 1.9, 3.1, 3.9, 5.1] * 4, dtype=float)


class TestReporterClassification:
    def test_returns_result_object(self):
        r = Reporter(TRUTHS_CLS, PREDS_CLS, run=0, epoch=0)
        assert isinstance(r.get_result(), Result)

    def test_metric_is_uar(self):
        r = Reporter(TRUTHS_CLS, PREDS_CLS, run=0, epoch=0)
        assert r.metric == "uar"
        assert r.METRIC == "UAR"

    def test_perfect_uar_near_one(self):
        r = Reporter(TRUTHS_CLS, PREDS_CLS, run=0, epoch=0)
        assert r.get_result().test == pytest.approx(1.0, abs=1e-6)

    def test_result_has_upper_lower(self):
        r = Reporter(TRUTHS_CLS, PREDS_CLS, run=0, epoch=0)
        result = r.get_result()
        assert hasattr(result, "upper")
        assert hasattr(result, "lower")

    def test_eer_metric_selected(self):
        glob_conf.config["MODEL"]["measure"] = "eer"
        # EER needs binary classes; use 0/1
        truths = np.array([0, 1] * 10)
        preds = np.array([0, 1] * 10)
        r = Reporter(truths, preds, run=0, epoch=0)
        assert r.metric == "eer"
        assert r.METRIC == "EER"

    def test_run_and_epoch_stored(self):
        r = Reporter(TRUTHS_CLS, PREDS_CLS, run=2, epoch=5)
        assert r.run == 2
        assert r.epoch == 5


class TestReporterRegression:
    def test_metric_is_mse_by_default(self):
        glob_conf.config["EXP"]["type"] = "regression"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert r.metric == "mse"
        assert r.METRIC == "MSE"

    def test_mse_close_to_zero_for_near_perfect(self):
        glob_conf.config["EXP"]["type"] = "regression"
        truths = np.array([1.0, 2.0, 3.0] * 5, dtype=float)
        preds = truths.copy()
        r = Reporter(truths, preds, run=0, epoch=0)
        assert r.get_result().test == pytest.approx(0.0, abs=1e-6)

    def test_mae_metric_selected(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "mae"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert r.metric == "mae"

    def test_ccc_metric_selected(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "ccc"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert r.metric == "ccc"

    def test_truths_cont_stored(self):
        glob_conf.config["EXP"]["type"] = "regression"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert hasattr(r, "truths_cont")


class TestReporterEmpty:
    def test_empty_truths_preds_no_crash(self):
        r = Reporter([], [], run=0, epoch=0)
        result = r.get_result()
        assert result.test == 0
