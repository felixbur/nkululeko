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

    def test_pcc_metric_selected(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "pcc"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert r.metric == "pcc"
        assert r.METRIC == "PCC"

    def test_pcc_near_one_for_near_perfect_correlation(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "pcc"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert r.get_result().test == pytest.approx(1.0, abs=1e-2)

    def test_truths_cont_stored(self):
        glob_conf.config["EXP"]["type"] = "regression"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        assert hasattr(r, "truths_cont")

    def test_truths_cont_stored_even_when_empty(self):
        """truths_cont/preds_cont must exist regardless of data length --
        callers like Experiment.plot_confmat_per_speaker() read them
        unconditionally for any non-classification report."""
        glob_conf.config["EXP"]["type"] = "regression"
        r = Reporter(np.array([]), np.array([]), run=0, epoch=0)
        assert hasattr(r, "truths_cont")
        assert hasattr(r, "preds_cont")


class TestPrintResultsAfterPlotConfmatrix:
    """Regression: runmanager.print_report() always calls
    plot_confmatrix() before print_results(). plot_confmatrix() binarizes
    self.truths/self.preds in place (for the confusion-matrix view), so
    print_results() must read the untouched truths_cont/preds_cont copies --
    otherwise the r_2/pcc it writes to the result file are computed on
    binarized 0/1 labels instead of the real continuous predictions."""

    def test_pcc_unaffected_by_prior_plot_confmatrix_call(self, tmp_path):
        from scipy.stats import pearsonr

        from nkululeko.reporting.report import Report

        glob_conf.config["EXP"]["type"] = "regression"
        r = Reporter(TRUTHS_REG, PREDS_REG, run=0, epoch=0)
        r.context.report = Report()
        expected_pcc = pearsonr(TRUTHS_REG, PREDS_REG)[0]

        # Call order matches runmanager.py's print_report(): plot_confmatrix
        # first (mutates self.truths/self.preds), then print_results.
        r.plot_confmatrix("order_bug_test", epoch=0)
        r.print_results(epoch=0, file_name="order_bug_test")

        res_dir = r.util.get_path("res_dir")
        with open(res_dir + "order_bug_test.txt") as f:
            content = f.read()
        written_pcc = float(content.split("pcc ")[1])
        assert written_pcc == pytest.approx(expected_pcc, abs=1e-3)


class TestReporterEmpty:
    def test_empty_truths_preds_no_crash(self):
        r = Reporter([], [], run=0, epoch=0)
        result = r.get_result()
        assert result.test == 0


class TestReporterClassificationReportMismatch:
    """classification_report raises ValueError when target_names doesn't
    match the classes actually present in truths/preds (e.g. a class never
    predicted). print_results must degrade gracefully instead of crashing.
    """

    def test_print_results_falls_back_without_crashing(self, tmp_path):
        glob_conf.config["EXP"]["root"] = str(tmp_path)
        truths = np.array([0, 1] * 5)
        preds = np.array([0, 1] * 5)
        r = Reporter(truths, preds, run=0, epoch=0)
        # print_results() prefers label_encoder.classes_ over context.labels
        # when a label_encoder is present; clear it so this test deterministically
        # exercises the context.labels path regardless of what other tests left
        # on the shared context.
        r.context.label_encoder = None
        # Claim more labels than actually appear in truths/preds so
        # classification_report raises ValueError.
        r.context.labels = ["a", "b", "c", "d", "e"]

        r.print_results(epoch=0, file_name="mismatch_test")

        res_dir = r.util.get_path("res_dir")
        with open(res_dir + "mismatch_test.txt") as f:
            content = f.read()
        assert "UAR" in content
