"""Unit tests for Featureset base class (nkululeko/feat_extract/featureset.py)."""

import configparser
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
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
def multiindex_data_df():
    """DataFrame with a (file, start, end) MultiIndex as used by audio feature extractors."""
    files = [f"audio_{i}.wav" for i in range(5)]
    starts = [pd.Timedelta(0)] * 5
    ends = [pd.Timedelta(seconds=1)] * 5
    idx = pd.MultiIndex.from_arrays([files, starts, ends], names=["file", "start", "end"])
    return pd.DataFrame({"label": range(5)}, index=idx)


@pytest.fixture
def featureset(data_df):
    return Featureset(name="test_feats", data_df=data_df, feats_type="os")


@pytest.fixture
def multiindex_featureset(multiindex_data_df):
    return Featureset(name="test_feats_mi", data_df=multiindex_data_df, feats_type="os")


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
        rng = np.random.default_rng(42)
        featureset.df = pd.DataFrame(
            rng.random((7, 3)), columns=["f1", "f2", "f3"], index=extra_idx
        )
        featureset.filter()
        assert len(featureset.df) == 5
        assert set(featureset.df.index).issubset(set(data_df.index))

    def test_filter_subset_selection_from_config(self, featureset, data_df):
        """When FEATS.features is set, filter() should select those columns."""
        glob_conf.config["FEATS"]["features"] = "['f1', 'f2']"
        rng = np.random.default_rng(42)
        featureset.df = pd.DataFrame(
            rng.random((5, 3)), columns=["f1", "f2", "f3"], index=data_df.index
        )
        featureset.filter()
        assert list(featureset.df.columns) == ["f1", "f2"]

    def test_filter_ignores_nonexistent_selected_features(
        self, featureset, data_df
    ):
        """Non-existent selected features are skipped, not raised."""
        glob_conf.config["FEATS"]["features"] = "['f1', 'ghost']"
        rng = np.random.default_rng(42)
        featureset.df = pd.DataFrame(
            rng.random((5, 2)), columns=["f1", "f2"], index=data_df.index
        )
        featureset.filter()
        assert "f1" in featureset.df.columns
        assert "ghost" not in featureset.df.columns


class TestExtractEmbeddingsWithErrorHandling:
    """Tests for Featureset._extract_embeddings_with_error_handling."""

    def _make_extract_fn(self, emb_dim=4, fail_indices=None):
        """Return an extract_fn that returns a fixed embedding or raises on given indices."""
        fail_indices = set(fail_indices or [])
        call_count = {"n": 0}

        def extract_fn(file, start, end):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx in fail_indices:
                raise RuntimeError(f"simulated failure for {file}")
            return np.ones(emb_dim, dtype=float)

        return extract_fn

    def test_all_succeed_returns_full_dataframe(self, multiindex_featureset, multiindex_data_df):
        """When no file fails, returned DataFrame has all rows."""
        extract_fn = self._make_extract_fn(emb_dim=4)
        result = multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(multiindex_data_df)
        assert result.shape[1] == 4

    def test_all_succeed_index_matches_data_df(self, multiindex_featureset, multiindex_data_df):
        """Returned DataFrame index matches data_df index when all succeed."""
        extract_fn = self._make_extract_fn(emb_dim=3)
        result = multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        pd.testing.assert_index_equal(result.index, multiindex_data_df.index)

    def test_failed_files_are_skipped(self, multiindex_featureset, multiindex_data_df):
        """Rows for files that raise exceptions are dropped from the result."""
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={1, 3})
        result = multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        # 5 files - 2 failures = 3 rows
        assert len(result) == 3

    def test_failed_file_indices_excluded(self, multiindex_featureset, multiindex_data_df):
        """The indices of failed files must not appear in the result."""
        all_index = multiindex_data_df.index.to_list()
        fail_indices = {0, 4}
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices=fail_indices)
        result = multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        expected_present = [all_index[i] for i in range(5) if i not in fail_indices]
        for idx in expected_present:
            assert idx in result.index
        for i in fail_indices:
            assert all_index[i] not in result.index

    def test_warn_issued_when_files_skipped(self, multiindex_featureset):
        """A warning must be issued when at least one file is skipped."""
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={2})
        with patch.object(multiindex_featureset.util, "warn") as mock_warn:
            multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        assert mock_warn.called

    def test_no_warn_when_all_succeed(self, multiindex_featureset):
        """No warning about skipped files when all extractions succeed."""
        extract_fn = self._make_extract_fn(emb_dim=4)
        with patch.object(multiindex_featureset.util, "warn") as mock_warn:
            multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        # warn may be called for other reasons but the "skipped N files" message should not appear
        skipped_calls = [
            call for call in mock_warn.call_args_list
            if "skipped" in str(call).lower() and "failed" in str(call).lower()
        ]
        assert len(skipped_calls) == 0

    def test_all_fail_returns_empty_dataframe(self, multiindex_featureset):
        """When every file fails, an empty DataFrame is returned."""
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={0, 1, 2, 3, 4})
        result = multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_embedding_values_correct(self, multiindex_featureset, multiindex_data_df):
        """Embeddings returned by extract_fn appear as rows in the result DataFrame."""
        rng = np.random.default_rng(0)
        embeddings = [rng.random(6) for _ in range(len(multiindex_data_df))]
        call_count = {"n": 0}

        def extract_fn(file, start, end):
            emb = embeddings[call_count["n"]]
            call_count["n"] += 1
            return emb

        result = multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        assert result.shape == (5, 6)
        for i in range(5):
            np.testing.assert_array_almost_equal(result.iloc[i].values, embeddings[i])
