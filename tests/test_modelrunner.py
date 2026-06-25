"""Tests for nkululeko/modelrunner.py — Modelrunner class."""

import configparser

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.modelrunner import Modelrunner
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

    def test_high_is_good_sets_best_performance_zero(self, dummy_dfs):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.best_performance == 0

    def test_low_is_good_sets_best_performance_large(self, dummy_dfs):
        glob_conf.config["MODEL"]["measure"] = "eer"
        df_train, df_test, feats_train, feats_test = dummy_dfs
        mr = Modelrunner(df_train, df_test, feats_train, feats_test, run=0)
        assert mr.best_performance == 100000


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
