"""Unit tests for Featureset base class (nkululeko/feat_extract/featureset.py)."""

import configparser

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


@pytest.fixture(autouse=True)
def setup_glob_conf():
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": "/tmp"}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb", "n_jobs": "1"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


@pytest.fixture
def data_df():
    idx = pd.RangeIndex(5)
    return pd.DataFrame({"file": [f"f{i}.wav" for i in range(5)]}, index=idx)


@pytest.fixture
def featureset(data_df):
    return Featureset(name="test_feats", data_df=data_df, feats_type="os")


class TestFeaturesetInit:
    def test_name_stored(self, featureset):
        assert featureset.name == "test_feats"

    def test_feats_type_stored(self, featureset):
        assert featureset.feats_type == "os"

    def test_data_df_stored(self, featureset, data_df):
        assert featureset.data_df is data_df

    def test_util_created(self, featureset):
        assert featureset.util is not None

    def test_n_jobs_from_config(self, featureset):
        assert featureset.n_jobs == 1


class TestFeaturesetExtract:
    def test_extract_is_noop_on_base(self, featureset):
        """Base class extract() should return None without error."""
        result = featureset.extract()
        assert result is None


class TestFeaturesetFilter:
    def test_filter_keeps_matching_index(self, featureset, data_df):
        """filter() should keep only rows whose index is in data_df."""
        # Attach a feature df with same index + 2 extra rows
        extra_idx = pd.RangeIndex(7)
        featureset.df = pd.DataFrame(
            np.random.rand(7, 3), columns=["f1", "f2", "f3"], index=extra_idx
        )
        featureset.filter()
        assert len(featureset.df) == 5
        assert set(featureset.df.index).issubset(set(data_df.index))

    def test_filter_subset_selection_from_config(self, featureset, data_df):
        """When FEATS.features is set, filter() should select those columns."""
        glob_conf.config["FEATS"]["features"] = "['f1', 'f2']"
        featureset.df = pd.DataFrame(
            np.random.rand(5, 3), columns=["f1", "f2", "f3"], index=data_df.index
        )
        featureset.filter()
        assert list(featureset.df.columns) == ["f1", "f2"]

    def test_filter_ignores_nonexistent_selected_features(
        self, featureset, data_df
    ):
        """Non-existent selected features are skipped, not raised."""
        glob_conf.config["FEATS"]["features"] = "['f1', 'ghost']"
        featureset.df = pd.DataFrame(
            np.random.rand(5, 2), columns=["f1", "f2"], index=data_df.index
        )
        featureset.filter()
        assert "f1" in featureset.df.columns
        assert "ghost" not in featureset.df.columns
