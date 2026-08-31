"""Tests for nkululeko/runmanager.py — Runmanager class."""

import configparser
import types

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.experiment_context import ExperimentContext, get_context, use_context
from nkululeko.runmanager import Runmanager
from nkululeko.utils.util import Util


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "test_run",
        "root": str(tmp_path),
        "runs": "1",
        "epochs": "1",
        "traindevtest": "False",
    }
    config["DATA"] = {"target": "emotion", "databases": "['test_db']"}
    config["MODEL"] = {"type": "xgb", "measure": "uar"}
    config["FEATS"] = {"type": "['os']"}
    config["PLOT"] = {}
    glob_conf.init_config(config)
    yield
    glob_conf.config = None


def _make_report(test_score, run=0, epoch=0):
    """Create a minimal mock report with result.test set."""
    result = types.SimpleNamespace(test=test_score)
    report = types.SimpleNamespace(result=result, run=run, epoch=epoch)
    return report


@pytest.fixture
def runmanager(tmp_path):
    """Construct a Runmanager with empty dataframes (no actual training)."""
    import pandas as pd

    df = pd.DataFrame({"emotion": []})
    feats = pd.DataFrame()
    return Runmanager(df, df, feats, feats)


class TestSearchBestResultAscending:
    def test_picks_highest_value(self, runmanager):
        reports = [
            _make_report(0.3),
            _make_report(0.7),
            _make_report(0.5),
        ]
        best = runmanager.search_best_result(reports, "ascending")
        assert best.result.test == pytest.approx(0.7)

    def test_returns_first_with_all_equal(self, runmanager):
        reports = [_make_report(0.5), _make_report(0.5)]
        best = runmanager.search_best_result(reports, "ascending")
        assert best.result.test == pytest.approx(0.5)

    def test_single_report(self, runmanager):
        reports = [_make_report(0.42)]
        best = runmanager.search_best_result(reports, "ascending")
        assert best.result.test == pytest.approx(0.42)


class TestSearchBestResultDescending:
    def test_picks_lowest_value(self, runmanager):
        reports = [
            _make_report(0.3),
            _make_report(0.7),
            _make_report(0.1),
        ]
        best = runmanager.search_best_result(reports, "descending")
        assert best.result.test == pytest.approx(0.1)

    def test_single_report(self, runmanager):
        reports = [_make_report(0.05)]
        best = runmanager.search_best_result(reports, "descending")
        assert best.result.test == pytest.approx(0.05)


class TestGetBestResult:
    def test_classification_uar_uses_ascending(self, runmanager):
        """UAR is higher-is-better, so get_best_result should return the max."""
        reports = [_make_report(0.2), _make_report(0.9), _make_report(0.5)]
        best = runmanager.get_best_result(reports)
        assert best.result.test == pytest.approx(0.9)

    def test_classification_eer_uses_descending(self, runmanager):
        """EER is lower-is-better."""
        glob_conf.config["MODEL"]["measure"] = "eer"
        reports = [_make_report(0.2), _make_report(0.9), _make_report(0.05)]
        best = runmanager.get_best_result(reports)
        assert best.result.test == pytest.approx(0.05)

    def test_regression_mse_uses_descending(self, runmanager):
        """MSE is lower-is-better."""
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "mse"
        reports = [_make_report(10.0), _make_report(0.5), _make_report(3.0)]
        best = runmanager.get_best_result(reports)
        assert best.result.test == pytest.approx(0.5)


class TestLoadModelContextPropagation:
    """Regression for the predict/reload path: when a pickled Runmanager
    (built during training, holding `context_train`) is reactivated under a
    fresh "live" ambient context (as `predict.py` sets up via a new
    `Experiment(config)`), `load_model()` must make its own context ambient
    for the whole call. Otherwise `EXP.run` gets updated on `context_train`
    but any object constructed deeper (e.g. a model's `Util()` with no
    explicit context) resolves `get_context()` to the stale, never-updated
    live context and looks for the model files under `run_0/` even when a
    later run (e.g. run 1) actually won.
    """

    def _make_context(self, run="0"):
        config = configparser.ConfigParser()
        config["EXP"] = {"run": run}
        config["MODEL"] = {"type": "xgb"}
        return ExperimentContext(config=config)

    def test_load_model_makes_own_context_ambient_for_nested_construction(self):
        context_train = self._make_context(run="0")
        context_live = self._make_context(run="0")

        runmgr = Runmanager.__new__(Runmanager)
        runmgr.context = context_train
        runmgr.util = Util("runmanager", context=context_train)

        observed_run = {}

        class _FakeModelrunner:
            def _select_model(self, model_type):
                # Stands in for e.g. MLP_Reg_model's constructor calling
                # `Util("mlp_reg")` with no explicit context, which falls
                # back to `get_context()`.
                observed_run["run"] = get_context().config["EXP"]["run"]
                return types.SimpleNamespace(load=lambda run, epoch: None)

        runmgr.modelrunner = _FakeModelrunner()
        report = types.SimpleNamespace(run=1, epoch=4)

        # Mimic predict.py: a freshly built Experiment made `context_live`
        # ambient, unrelated to the reloaded, training-time `context_train`.
        # Scoped with `use_context` (the codebase's own idiom) so the ambient
        # context is restored via its token even if load_model raises.
        with use_context(context_live):
            runmgr.load_model(report)

        # load_model's set_config_val("EXP", "run", 1) must be visible to
        # the nested _select_model call via the ambient context.
        assert observed_run["run"] == "1"
        # And it must not have leaked into the unrelated live context.
        assert context_live.config["EXP"]["run"] == "0"

    def test_load_model_restores_prior_ambient_context_afterwards(self):
        context_train = self._make_context(run="0")
        context_live = self._make_context(run="0")

        runmgr = Runmanager.__new__(Runmanager)
        runmgr.context = context_train
        runmgr.util = Util("runmanager", context=context_train)
        runmgr.modelrunner = types.SimpleNamespace(
            _select_model=lambda model_type: types.SimpleNamespace(
                load=lambda run, epoch: None
            )
        )

        with use_context(context_live):
            runmgr.load_model(types.SimpleNamespace(run=1, epoch=4))
            # Ambient context after the call must be back to whatever it was
            # before -- load_model must not leave context_train installed
            # globally. Asserted inside the `with` so this checks the
            # use_context scope load_model was invoked in, not some later,
            # unrelated ambient state.
            assert get_context() is context_live


class TestRunmanagerInit:
    def test_stores_split3_false(self, runmanager):
        assert runmanager.split3 is False

    def test_target_from_config(self, runmanager):
        assert runmanager.target == "emotion"

    def test_split3_true_when_configured(self, tmp_path):
        import pandas as pd

        glob_conf.config["EXP"]["traindevtest"] = "True"
        df = pd.DataFrame({"emotion": []})
        feats = pd.DataFrame()
        rm = Runmanager(df, df, feats, feats)
        assert rm.split3 is True


class TestDoRunsSplit3ReportsRealTestResult:
    """Regression: under [EXP] traindevtest=True (split3), do_runs() must
    report the real test-set result, not the dev-phase result used only for
    checkpoint selection.

    do_runs() ran the dev-monitoring loop, appended its best dev report to
    self.best_results, then separately computed self.test_report by
    re-evaluating the best model on the real test set - but never fed that
    test_report back into self.best_results. Since experiment.py's final
    summary reads exactly self.best_results (`self.reports =
    self.runmgr.best_results`), every traindevtest=True experiment's printed
    "final" score was silently the dev score, for every model type.
    """

    def test_best_results_holds_test_report_not_dev_report(self, tmp_path, monkeypatch):
        import pandas as pd

        glob_conf.config["EXP"]["traindevtest"] = "True"
        glob_conf.config["EXP"]["runs"] = "1"
        df = pd.DataFrame({"emotion": []})
        feats = pd.DataFrame()
        rm = Runmanager(df, df, feats, feats, dev_x=df, dev_y=feats)

        dev_report = _make_report(0.90, run=0, epoch=1)
        test_report = _make_report(0.25, run=0, epoch=1)

        class FakeModel:
            def load(self, run, epoch):
                pass

        class FakeModelrunner:
            def __init__(self, *args, **kwargs):
                pass

            def do_epochs(self):
                return [dev_report], 1

            def _select_model(self, model_type):
                return FakeModel()

            def eval_specific_model(self, model, df_test, feats_test, split_name=None):
                return test_report

        monkeypatch.setattr("nkululeko.runmanager.Modelrunner", FakeModelrunner)
        monkeypatch.setattr(Runmanager, "print_report", lambda self, r, p: None)

        rm.do_runs()

        assert rm.best_results == [test_report]
        assert rm.best_results[0] is not dev_report
