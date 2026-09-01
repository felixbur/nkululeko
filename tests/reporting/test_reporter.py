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

    def test_truths_cont_is_a_copy_not_an_alias(self):
        """truths_cont/preds_cont must survive in-place mutation of
        self.truths/self.preds, not just reassignment (continuous_to_categorical
        happens to reassign today, which is why an alias worked, but that's
        incidental -- anything that mutates in place, or mutates the
        caller's original arrays, must not affect the stored copies)."""
        glob_conf.config["EXP"]["type"] = "regression"
        r = Reporter(TRUTHS_REG.copy(), PREDS_REG.copy(), run=0, epoch=0)
        r.truths[:] = 0
        r.preds[:] = 0
        assert np.array_equal(r.truths_cont, TRUTHS_REG)
        assert np.array_equal(r.preds_cont, PREDS_REG)


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


class TestReporterBinarySensitivitySpecificity:
    """Sensitivity/specificity should be reported automatically for binary
    classification (issue #420), and absent for multi-class tasks."""

    def test_perfect_binary_predictions(self, tmp_path, caplog):
        truths = np.array([0, 1] * 10)
        preds = np.array([0, 1] * 10)
        r = Reporter(truths, preds, run=0, epoch=0)
        # context is a shared singleton across tests; pin it explicitly.
        r.context.label_encoder = None
        r.context.labels = ["0", "1"]
        with caplog.at_level("DEBUG"):
            r.print_results(epoch=0, file_name="binary_perfect")

        res_dir = r.util.get_path("res_dir")
        with open(res_dir + "binary_perfect.txt") as f:
            content = f.read()
        assert '"sensitivity": 1.0' in content
        assert '"specificity": 1.0' in content
        assert "sensitivity ('1'): 1.0000" in content
        assert "specificity ('0'): 1.0000" in content

        # Also logged to the DEBUG log, not just written to file.
        assert "sensitivity ('1'): 1.0000" in caplog.text
        assert "specificity ('0'): 1.0000" in caplog.text

    def test_imperfect_binary_predictions(self, tmp_path):
        # class 0 (negative): 5 samples, 1 misclassified as 1 -> specificity 4/5
        # class 1 (positive): 5 samples, 1 misclassified as 0 -> sensitivity 4/5
        truths = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        preds = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        r = Reporter(truths, preds, run=0, epoch=0)
        r.context.label_encoder = None
        r.context.labels = ["0", "1"]
        r.print_results(epoch=0, file_name="binary_imperfect")

        res_dir = r.util.get_path("res_dir")
        with open(res_dir + "binary_imperfect.txt") as f:
            content = f.read()
        assert '"sensitivity": 0.8' in content
        assert '"specificity": 0.8' in content

    def test_multiclass_has_no_sensitivity_specificity(self, tmp_path):
        r = Reporter(TRUTHS_CLS, PREDS_CLS, run=0, epoch=0)
        r.context.label_encoder = None
        r.context.labels = ["0", "1", "2"]
        r.print_results(epoch=0, file_name="multiclass_no_sens_spec")

        res_dir = r.util.get_path("res_dir")
        with open(res_dir + "multiclass_no_sens_spec.txt") as f:
            content = f.read()
        assert "sensitivity" not in content
        assert "specificity" not in content

    def test_positive_class_follows_data_labels_order_not_encoder_alphabetization(
        self, tmp_path
    ):
        """label_encoder.classes_ is always alphabetically sorted by sklearn,
        which carries no domain meaning (e.g. "anger" < "neutral"). The
        positive/negative choice must follow the order the user wrote in
        DATA.labels (context.labels), not the encoder's alphabetical order.
        """
        # Simulate label_encoder encoding classes alphabetically: 0="anger", 1="neutral".
        truths = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # anger (0): 3/5 correctly predicted (recall 0.6)
        # neutral (1): 5/5 correctly predicted (recall 1.0)
        preds = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        r = Reporter(truths, preds, run=0, epoch=0)
        r.context.label_encoder = type(
            "FakeEncoder", (), {"classes_": np.array(["anger", "neutral"])}
        )()
        # User wrote DATA.labels = ["neutral", "anger"]: neutral=negative, anger=positive.
        r.context.labels = ["neutral", "anger"]
        r.print_results(epoch=0, file_name="positive_class_order")

        res_dir = r.util.get_path("res_dir")
        with open(res_dir + "positive_class_order.txt") as f:
            content = f.read()
        assert "sensitivity ('anger'): 0.6000" in content
        assert "specificity ('neutral'): 1.0000" in content
        assert '"sensitivity": 0.6' in content
        assert '"specificity": 1.0' in content


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
