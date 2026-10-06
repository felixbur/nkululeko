"""Tests for nkululeko/modelrunner.py — Modelrunner class."""

import configparser
import os

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.modelrunner import Modelrunner, validate_model_task_support
from nkululeko.utils.errors import NkululukoError


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "test_mr",
        "root": str(tmp_path),
        "runs": "1",
        "epochs": "1",
        "traindevtest": "False",
    }
    config["DATA"] = {"target": "emotion", "databases": "['test_db']"}
    config["MODEL"] = {"type": "xgb", "measure": "uar"}
    config["FEATS"] = {"type": "['os']", "balancing": ""}
    config["PLOT"] = {}
    glob_conf.init_config(config)
    yield
    glob_conf.config = None


@pytest.fixture
def dummy_dfs():
    rng = np.random.default_rng(0)
    df_train = pd.DataFrame({"emotion": [0, 1, 0, 1]})
    df_test = pd.DataFrame({"emotion": [0, 1]})
    feats_train = pd.DataFrame(rng.random((4, 3)))
    feats_test = pd.DataFrame(rng.random((2, 3)))
    return df_train, df_test, feats_train, feats_test


class TestModelrunnerInit:
    def test_run_stored(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=2)
        assert mr.run == 2

    def test_target_from_config(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.target == "emotion"

    def test_split_name_uppercase(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(
            df_train, df_test, feats_train, feats_test, run=0, split_name="dev"
        )
        assert mr.split_name == "DEV"

    def test_default_split_name_is_test(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.split_name == "TEST"

    def test_best_performance_unset_until_first_epoch(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.best_performance is None

    def test_first_score_is_always_best_for_either_direction(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr._is_better(-5.0)  # high-is-good (uar), no baseline yet
        glob_conf.config["MODEL"]["measure"] = "eer"
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.best_performance is None
        assert mr._is_better(99999.0)  # low-is-good (eer), no baseline yet


class TestSelectModel:
    def _make_mr(self, model_type, dummy_dfs):
        glob_conf.config["MODEL"]["type"] = model_type
        df_train, df_test, feats_train, feats_test = dummy_dfs
        return Modelrunner(df_train, df_test, feats_train, feats_test, run=0)

    def test_xgb_model_selected(self, dummy_dfs):
        from nkululeko.models.model_xgb import XGB_model

        mr = self._make_mr("xgb", dummy_dfs)
        assert isinstance(mr.model, XGB_model)

    def test_svm_model_selected(self, dummy_dfs):
        from nkululeko.models.model_svm import SVM_model

        mr = self._make_mr("svm", dummy_dfs)
        assert isinstance(mr.model, SVM_model)

    def test_knn_model_selected(self, dummy_dfs):
        from nkululeko.models.model_knn import KNN_model

        mr = self._make_mr("knn", dummy_dfs)
        assert isinstance(mr.model, KNN_model)

    def test_aasist_model_selected(self, dummy_dfs, monkeypatch):
        # AasistModel loads an SSL checkpoint in __init__, so patch in a
        # stand-in; this only checks that "aasist" is routed to the class.
        import nkululeko.models.model_aasist as model_aasist

        class FakeAasist:
            is_classifier = True
            is_regressor = False

            def __init__(self, df_train, df_test, feats_train, feats_test):
                self.df_train = df_train

        monkeypatch.setattr(model_aasist, "AasistModel", FakeAasist)

        mr = self._make_mr("aasist", dummy_dfs)

        assert isinstance(mr.model, FakeAasist)

    def test_bayes_model_selected(self, dummy_dfs):
        from nkululeko.models.model_bayes import Bayes_model

        mr = self._make_mr("bayes", dummy_dfs)
        assert isinstance(mr.model, Bayes_model)

    def test_tree_model_selected(self, dummy_dfs):
        from nkululeko.models.model_tree import Tree_model

        mr = self._make_mr("tree", dummy_dfs)
        assert isinstance(mr.model, Tree_model)

    def test_gmm_model_selected(self, dummy_dfs):
        from nkululeko.models.model_gmm import GMM_model

        mr = self._make_mr("gmm", dummy_dfs)
        assert isinstance(mr.model, GMM_model)

    def test_unknown_model_raises(self, dummy_dfs):
        glob_conf.config["MODEL"]["type"] = "not_a_model"
        df_train, df_test, feats_train, feats_test = dummy_dfs
        with pytest.raises(NkululukoError):
            Modelrunner(df_train, df_test, feats_train, feats_test, run=0)

    def test_regression_model_for_regression_exp(self, dummy_dfs):
        from nkululeko.models.model_svr import SVR_model

        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["type"] = "svr"
        glob_conf.config["MODEL"]["measure"] = "mse"
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert isinstance(mr.model, SVR_model)

    def test_classifier_model_with_regression_exp_raises(self, dummy_dfs):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["type"] = "svm"
        glob_conf.config["MODEL"]["measure"] = "mse"
        df_train, df_test, feats_train, feats_test = dummy_dfs
        with pytest.raises(NkululukoError):
            Modelrunner(df_train, df_test, feats_train, feats_test, run=0)

    def test_regressor_model_with_classification_exp_raises(self, dummy_dfs):
        glob_conf.config["EXP"]["type"] = "classification"
        glob_conf.config["MODEL"]["type"] = "svr"
        df_train, df_test, feats_train, feats_test = dummy_dfs
        with pytest.raises(NkululukoError):
            Modelrunner(df_train, df_test, feats_train, feats_test, run=0)


class _CapabilityStub:
    """Minimal stand-in exposing is_classifier/is_regressor capability flags.

    Used to exercise ``validate_model_task_support`` without instantiating
    heavy real models.
    """

    def __init__(self, is_classifier=None, is_regressor=None):
        self.is_classifier = is_classifier
        self.is_regressor = is_regressor


class TestValidateModelTaskSupport:
    """Unit tests for validate_model_task_support capability-flag handling."""

    def test_explicit_regressor_allows_svm_regression(self):
        """A model declaring is_regressor=True may run a regression task even
        when its model_type ('svm') is normally a classifier type."""
        model = _CapabilityStub(is_classifier=False, is_regressor=True)
        validate_model_task_support(model_type="svm", task="regression", model=model)

    def test_explicit_classifier_allows_svr_classification(self):
        """A model declaring is_classifier=True may run a classification task
        even when its model_type ('svr') is normally a regressor type."""
        model = _CapabilityStub(is_classifier=True, is_regressor=False)
        validate_model_task_support(
            model_type="svr", task="classification", model=model
        )

    def test_classifier_false_raises_for_classification(self):
        """A model with is_classifier=False must not be used for
        classification, even if its model_type ('svm') is a classifier type."""
        model = _CapabilityStub(is_classifier=False, is_regressor=True)
        with pytest.raises(NkululukoError):
            validate_model_task_support(
                model_type="svm", task="classification", model=model
            )

    def test_regressor_false_raises_for_regression(self):
        """A model with is_regressor=False must not be used for regression,
        even if its model_type ('svr') is a regressor type."""
        model = _CapabilityStub(is_classifier=True, is_regressor=False)
        with pytest.raises(NkululukoError):
            validate_model_task_support(
                model_type="svr", task="regression", model=model
            )


class TestCheckFeatureBalancing:
    def test_no_balancing_config_no_change(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.feats_train.shape == feats_train.shape

    def test_balancing_applied_when_configured(self, dummy_dfs):
        """Oversampling should not reduce the training set size."""
        glob_conf.config["FEATS"]["balancing"] = "ros"
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        # After balancing df_train and feats_train sizes must still match
        assert mr.df_train.shape[0] == mr.feats_train.shape[0]


class TestEvalSpecificModel:
    def test_split_name_restored_after_eval(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(
            df_train, df_test, feats_train, feats_test, run=0, split_name="dev"
        )
        original = mr.split_name

        class FakeModel:
            store_path = "fake"

            def reset_test(self, df, feats):
                pass  # no-op stub: test only verifies split_name is restored

            def predict(self):
                import types

                result = types.SimpleNamespace(test=0.5)
                r = types.SimpleNamespace(result=result)
                r.set_id = lambda run, epoch: None
                return r

        fake = FakeModel()
        mr.eval_specific_model(fake, df_test, feats_test, split_name="test")
        assert mr.split_name == original


class TestEmptyTestSetGuard:
    """An empty (0-row) test/dev split previously crashed deep inside
    sklearn's predict()/predict_proba() (0-sample arrays are rejected there
    before Reporter's own empty-truths/preds guard is ever reached).
    Modelrunner must skip prediction entirely and report a trivial zero
    result instead."""

    def test_eval_specific_model_skips_predict_for_empty_test_set(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)

        class FakeModel:
            store_path = "fake"

            def reset_test(self, df, feats):
                pass

            def predict(self):
                raise AssertionError("predict() must not be called for an empty test set")

        report = mr.eval_specific_model(
            FakeModel(), df_test.iloc[0:0], feats_test.iloc[0:0], split_name="test"
        )

        assert report.result.test == 0
        assert report.probas is None

    def test_eval_last_model_skips_predict_for_empty_test_set(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        mr.model.reset_test = lambda df, feats: None

        def _fail_predict():
            raise AssertionError("predict() must not be called for an empty test set")

        mr.model.predict = _fail_predict

        report = mr.eval_last_model(df_test.iloc[0:0], feats_test.iloc[0:0])

        assert report.result.test == 0
        assert report.probas is None

    def test_do_epochs_skips_predict_for_empty_test_set(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(
            df_train, df_test.iloc[0:0], feats_train, feats_test.iloc[0:0], run=0
        )

        def _fail_predict():
            raise AssertionError("predict() must not be called for an empty test set")

        mr.model.predict = _fail_predict

        reports, epoch = mr.do_epochs()

        assert len(reports) == 1
        assert reports[0].result.test == 0


class TestEmptyFeatsTestClassicModel:
    """Regression: the df_test-only guard added for GH #441 must not drop
    the original protection for classic (non-finetuned) models, whose
    predict() consumes feats_test directly (model.py's get_predictions()),
    not df_test. A desynced empty feats_test with a non-empty df_test must
    still be caught, even though that combination is rare in practice
    (datasplitter.py normally keeps the two in sync)."""

    def test_do_epochs_skips_predict_when_feats_test_empty_but_df_test_nonempty(
        self, dummy_dfs
    ):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test.iloc[0:0], run=0)

        def _fail_predict():
            raise AssertionError(
                "predict() must not be called when feats_test is empty"
            )

        mr.model.predict = _fail_predict

        reports, epoch = mr.do_epochs()

        assert len(reports) == 1
        assert reports[0].result.test == 0

    def test_eval_specific_model_skips_predict_when_feats_test_empty_but_df_test_nonempty(
        self, dummy_dfs
    ):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)

        class FakeModel:
            store_path = "fake"

            def reset_test(self, df, feats):
                pass

            def predict(self):
                raise AssertionError(
                    "predict() must not be called when feats_test is empty"
                )

        report = mr.eval_specific_model(
            FakeModel(), df_test, feats_test.iloc[0:0], split_name="test"
        )

        assert report.result.test == 0
        assert report.probas is None


class TestFinetunedNoneFeatsTest:
    """GH #441: finetuned models have no feature matrix (FEATS.type = [],
    so feats_test is None, not an empty DataFrame). The empty-test-set
    guards used to check len(feats_test)/len(self.feats_test), which
    crashed with TypeError: object of type 'NoneType' has no len() as soon
    as training finished on a non-empty test set. The guards must check the
    always-present df_test instead."""

    def test_do_epochs_finetuned_none_feats_test_nonempty_df(self, dummy_dfs):
        df_train, df_test, feats_train, _feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, None, run=0)

        from nkululeko.reporting.reporter import Reporter

        class FakeFinetunedModel:
            model_type = "finetuned"
            store_path = "fake"

            def is_ann(self):
                return True

            def train(self):
                pass

            def predict(self):
                return Reporter(
                    df_test["emotion"].to_numpy(),
                    df_test["emotion"].to_numpy(),
                    0,
                    1,
                    context=mr.context,
                )

        mr.model = FakeFinetunedModel()

        # Must not raise TypeError: object of type 'NoneType' has no len()
        reports, epoch = mr.do_epochs()

        assert len(reports) == 1
        assert reports[0].result.test != 0

    def test_eval_last_model_finetuned_none_feats_test_nonempty_df(self, dummy_dfs):
        df_train, df_test, feats_train, _feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, None, run=0)
        mr.model.reset_test = lambda df, feats: None

        from nkululeko.reporting.reporter import Reporter

        mr.model.predict = lambda: Reporter(
            df_test["emotion"].to_numpy(),
            df_test["emotion"].to_numpy(),
            0,
            1,
            context=mr.context,
        )

        # Must not raise TypeError: object of type 'NoneType' has no len()
        report = mr.eval_last_model(df_test, None)

        assert report.result.test != 0

    def test_eval_specific_model_finetuned_none_feats_test_nonempty_df(self, dummy_dfs):
        df_train, df_test, feats_train, _feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, None, run=0)

        from nkululeko.reporting.reporter import Reporter

        class FakeModel:
            store_path = "fake"

            def reset_test(self, df, feats):
                pass

            def predict(self):
                return Reporter(
                    df_test["emotion"].to_numpy(),
                    df_test["emotion"].to_numpy(),
                    0,
                    1,
                    context=mr.context,
                )

        # Must not raise TypeError: object of type 'NoneType' has no len()
        report = mr.eval_specific_model(FakeModel(), df_test, None, split_name="test")

        assert report.result.test != 0


class _FakeResult:
    metric = "uar"

    def __init__(self, v):
        self.v = v

    def get_test_result(self):
        # Match the real Result.get_test_result()'s .3f rounding: near-tied
        # raw scores can come out identical here even though they aren't.
        return f"test: {self.v:.3f} {self.metric}"

    def get_result(self):
        return self.v


class _FakeReport:
    def __init__(self, v):
        self.result = _FakeResult(v)

    def get_result(self):
        return self.result

    def set_id(self, run, epoch):
        pass

    def plot_confmatrix(self, *a, **k):
        pass


class TestSaveFlag:
    """[MODEL] save falls back to [EXP] save; best checkpoint kept when off (#67)."""

    SCORES = [0.5, 0.8, 0.6]  # best is epoch 1

    def _run(self, dummy_dfs, tmp_path, exp_save=None, model_save=None, only_test=None):
        """Run 3 fake epochs; return names of checkpoint files left on disk."""
        df_train, df_test, feats_train, feats_test = dummy_dfs
        cfg = glob_conf.config
        cfg["EXP"]["epochs"] = "3"
        if exp_save is not None:
            cfg["EXP"]["save"] = exp_save
        if model_save is not None:
            cfg["MODEL"]["save"] = model_save
        if only_test is not None:
            cfg["MODEL"]["only_test"] = only_test
        mr = Modelrunner(
            df_train, df_test.iloc[0:0], feats_train, feats_test.iloc[0:0], run=0
        )
        scores = iter(self.SCORES)

        def store():
            path = str(tmp_path / f"m_{mr.model.epoch}.model")
            for f in [path] + mr.model.sidecar_paths(path):
                (tmp_path / os.path.basename(f)).write_text("x")
            mr.model.store_path = path

        mr.model.store_path = str(tmp_path / "m.model")
        mr.model.is_ann = lambda: True
        mr.model.train = lambda: None
        mr.model.load = lambda run, epoch: mr.model.set_id(run, epoch)
        mr.model.reset_test = lambda df, feats: None
        mr.model.store = store
        mr.model.predict = lambda: _FakeReport(next(scores))
        mr.model.set_id = lambda run, epoch: setattr(mr.model, "epoch", epoch)
        mr._is_empty_split = lambda *a: False
        mr.do_epochs()
        return sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("m_"))

    @staticmethod
    def _files(*epochs):
        return sorted(
            f"m_{e}.model{ext}" for e in epochs for ext in ("", ".sha256")
        )

    def test_negative_scores_keep_best_checkpoint(self, dummy_dfs, tmp_path):
        """pcc/ccc can be negative: the best epoch must still be retained."""
        self.SCORES = [-0.9, -0.2, -0.5]  # best is epoch 1
        assert self._run(dummy_dfs, tmp_path, model_save="False") == self._files(1)

    def test_default_stores_every_epoch(self, dummy_dfs, tmp_path):
        assert self._run(dummy_dfs, tmp_path) == self._files(0, 1, 2)

    def test_exp_save_false_keeps_only_best(self, dummy_dfs, tmp_path):
        assert self._run(dummy_dfs, tmp_path, exp_save="False") == self._files(1)

    def test_model_save_false_keeps_only_best(self, dummy_dfs, tmp_path):
        assert self._run(dummy_dfs, tmp_path, model_save="False") == self._files(1)

    def test_model_save_overrides_exp_save(self, dummy_dfs, tmp_path):
        got = self._run(dummy_dfs, tmp_path, exp_save="False", model_save="True")
        assert got == self._files(0, 1, 2)

    def test_only_test_does_not_prune_existing_checkpoints(self, dummy_dfs, tmp_path):
        for e in (0, 1, 2):
            (tmp_path / f"m_{e}.model").write_text("x")
        got = self._run(dummy_dfs, tmp_path, exp_save="False", only_test="True")
        assert got == sorted(f"m_{e}.model" for e in (0, 1, 2))

    def test_only_test_false_string_is_not_truthy(self, dummy_dfs, tmp_path):
        """An explicit `only_test = False` (a string in the ini) must not
        disable checkpoint retention."""
        got = self._run(dummy_dfs, tmp_path, exp_save="False", only_test="False")
        assert got == self._files(1)

    def test_near_tied_rounded_scores_pick_true_raw_best(self, dummy_dfs, tmp_path):
        """GH #450 review: best-epoch selection must compare raw scores,
        not the .3f-rounded display string -- two epochs whose raw scores
        differ but round to the same value must not tie, since
        runmanager.search_best_result compares the raw floats to decide
        which epoch to reload later."""
        self.SCORES = [0.85499, 0.85501, 0.3]  # both round to 0.855; epoch 1 wins
        assert self._run(dummy_dfs, tmp_path, model_save="False") == self._files(1)

    def test_patience_baseline_not_beaten_by_negative_first_epoch(
        self, dummy_dfs, tmp_path
    ):
        """GH #450 review: patience's initial sentinel (`highest = 0`) must
        not block a legitimately-improving negative-valued metric run (e.g.
        pcc/ccc) from ever registering an improvement. With the old
        `highest = 0` sentinel and high_is_good() == True, no negative
        score can ever beat it, so patience_counter increments from epoch 0
        and can trigger early stopping even while scores strictly improve."""
        self.SCORES = [-0.5, -0.3, -0.1]  # strictly improving, all negative
        glob_conf.config["MODEL"]["patience"] = "1"
        got = self._run(dummy_dfs, tmp_path, model_save="True")
        assert got == self._files(0, 1, 2)

    def test_remove_checkpoint_deletes_adm_sidecars(self, dummy_dfs, tmp_path):
        from nkululeko.models.model_adm import ADMModel

        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        mr.model.sidecar_paths = ADMModel.sidecar_paths
        path = str(tmp_path / "m.model")
        files = [path] + ADMModel.sidecar_paths(path)
        for f in files:
            open(f, "w").write("x")
        mr._remove_checkpoint(path)
        assert not any(os.path.exists(f) for f in files)
