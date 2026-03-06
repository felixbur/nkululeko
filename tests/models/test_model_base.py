"""Unit tests for Model base class (nkululeko/models/model.py)."""

import configparser
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb", "n_jobs": "2"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


@pytest.fixture
def dummy_dfs():
    df_train = pd.DataFrame({"emotion": ["happy", "sad", "angry", "happy"]})
    df_test = pd.DataFrame({"emotion": ["sad", "angry"]})
    rng = np.random.default_rng(42)
    feats_train = pd.DataFrame(rng.random((4, 5)))
    feats_test = pd.DataFrame(rng.random((2, 5)))
    return df_train, df_test, feats_train, feats_test


@pytest.fixture
def model(dummy_dfs):
    df_train, df_test, feats_train, feats_test = dummy_dfs
    return Model(df_train, df_test, feats_train, feats_test)


class TestModelInit:
    def test_name_undefined(self, model):
        assert model.name == "undefined"

    def test_target_set_from_config(self, model):
        assert model.target == "emotion"

    def test_model_type_classic(self, model):
        assert model.model_type == "classic"

    def test_run_and_epoch_zero(self, model):
        assert model.run == 0
        assert model.epoch == 0

    def test_n_jobs_from_config(self, model):
        assert model.n_jobs == 2

    def test_dataframes_stored(self, dummy_dfs, model):
        df_train, df_test, feats_train, feats_test = dummy_dfs
        assert model.df_train is df_train
        assert model.df_test is df_test
        assert model.feats_train is feats_train
        assert model.feats_test is feats_test


class TestModelIsAnn:
    def test_classic_is_not_ann(self, model):
        assert model.is_ann() is False

    def test_ann_type_is_ann(self, model):
        model.model_type = "ann"
        assert model.is_ann() is True

    def test_finetuned_is_ann(self, model):
        model.model_type = "finetuned"
        assert model.is_ann() is True


class TestModelSetModelType:
    def test_sets_model_type(self, model):
        model.set_model_type("ann")
        assert model.model_type == "ann"


class TestModelSetTestdata:
    def test_replaces_test_data(self, model):
        new_df = pd.DataFrame({"emotion": ["happy"]})
        rng = np.random.default_rng(42)
        new_feats = pd.DataFrame(rng.random((1, 5)))
        model.set_testdata(new_df, new_feats)
        assert model.df_test is new_df
        assert model.feats_test is new_feats


class TestModelResetTest:
    def test_reset_test(self, model):
        new_df = pd.DataFrame({"emotion": ["angry"]})
        rng = np.random.default_rng(42)
        new_feats = pd.DataFrame(rng.random((1, 5)))
        model.reset_test(new_df, new_feats)
        assert model.df_test is new_df
        assert model.feats_test is new_feats


class TestModelSetId:
    def test_run_epoch_stored(self, model, tmp_path):
        glob_conf.config["EXP"]["root"] = str(tmp_path)
        glob_conf.config["EXP"]["run"] = "0"
        model.util.config = glob_conf.config
        model.set_id(1, 3)
        assert model.run == 1
        assert model.epoch == 3
        assert "001_003" in model.store_path or "1_003" in model.store_path
