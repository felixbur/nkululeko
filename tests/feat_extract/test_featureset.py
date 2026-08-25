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
    files = [f"f{i}.wav" for i in range(5)]
    return pd.DataFrame({"label": range(5)}, index=pd.Index(files, name="file"))


@pytest.fixture
def multiindex_data_df():
    """DataFrame with a (file, start, end) MultiIndex as used by audio feature extractors."""
    files = [f"audio_{i}.wav" for i in range(5)]
    starts = [pd.Timedelta(0)] * 5
    ends = [pd.Timedelta(seconds=1)] * 5
    idx = pd.MultiIndex.from_arrays(
        [files, starts, ends], names=["file", "start", "end"]
    )
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


class TestNeedsExtraction:
    """Direct coverage for _needs_extraction, which delegates to the shared
    should_reuse_file util -- previously only ever exercised indirectly
    (through a subclass's extract()), and several subclasses duplicated its
    logic ad-hoc instead of calling it at all."""

    def test_missing_storage_needs_extraction(self, featureset, tmp_path):
        storage = str(tmp_path / "missing.pkl")
        assert featureset._needs_extraction(storage) is True

    def test_existing_storage_does_not_need_extraction(self, featureset, tmp_path):
        storage = tmp_path / "cached.pkl"
        storage.write_text("")
        assert featureset._needs_extraction(str(storage)) is False

    def test_no_reuse_forces_extraction_even_if_cached(self, featureset, tmp_path):
        storage = tmp_path / "cached.pkl"
        storage.write_text("")
        glob_conf.config["FEATS"]["no_reuse"] = "True"
        assert featureset._needs_extraction(str(storage)) is True

    def test_needs_feature_extraction_flag_forces_extraction(self, featureset, tmp_path):
        storage = tmp_path / "cached.pkl"
        storage.write_text("")
        glob_conf.config["FEATS"]["needs_feature_extraction"] = "True"
        assert featureset._needs_extraction(str(storage)) is True


class TestFeaturesetFilter:
    def test_filter_keeps_matching_index(self, featureset, data_df):
        """filter() should keep only rows whose index is in data_df."""
        # Attach a feature df with same index + 2 extra rows
        extra_files = [f"f{i}.wav" for i in range(7)]
        extra_idx = pd.Index(extra_files, name="file")
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

    def test_filter_ignores_nonexistent_selected_features(self, featureset, data_df):
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

    def test_all_succeed_returns_full_dataframe(
        self, multiindex_featureset, multiindex_data_df
    ):
        """When no file fails, returned DataFrame has all rows."""
        extract_fn = self._make_extract_fn(emb_dim=4)
        result = multiindex_featureset._extract_embeddings_with_error_handling(
            extract_fn
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(multiindex_data_df)
        assert result.shape[1] == 4

    def test_all_succeed_index_matches_data_df(
        self, multiindex_featureset, multiindex_data_df
    ):
        """Returned DataFrame index matches data_df index when all succeed."""
        extract_fn = self._make_extract_fn(emb_dim=3)
        result = multiindex_featureset._extract_embeddings_with_error_handling(
            extract_fn
        )
        pd.testing.assert_index_equal(result.index, multiindex_data_df.index)

    def test_failed_files_are_skipped(self, multiindex_featureset, multiindex_data_df):
        """Rows for files that raise exceptions are dropped from the result."""
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={1, 3})
        result = multiindex_featureset._extract_embeddings_with_error_handling(
            extract_fn
        )
        # 5 files - 2 failures = 3 rows
        assert len(result) == 3

    def test_failed_file_indices_excluded(
        self, multiindex_featureset, multiindex_data_df
    ):
        """The indices of failed files must not appear in the result."""
        all_index = multiindex_data_df.index.to_list()
        fail_indices = {0, 4}
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices=fail_indices)
        result = multiindex_featureset._extract_embeddings_with_error_handling(
            extract_fn
        )
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
            call
            for call in mock_warn.call_args_list
            if "skipped" in str(call).lower() and "failed" in str(call).lower()
        ]
        assert len(skipped_calls) == 0

    def test_all_fail_aborts_when_exceeding_threshold(self, multiindex_featureset):
        """When all files fail and rate exceeds threshold, util.error() is called."""
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={0, 1, 2, 3, 4})
        with patch.object(multiindex_featureset.util, "error") as mock_error:
            # Prevent sys.exit by mocking error
            mock_error.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                multiindex_featureset._extract_embeddings_with_error_handling(
                    extract_fn
                )
        assert mock_error.called
        assert "exceeds" in str(mock_error.call_args).lower()

    def test_below_threshold_no_abort(self, multiindex_featureset):
        """When failure rate is below threshold, no abort occurs."""
        # 1 out of 5 fails = 20%, default threshold is 50%
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={2})
        with patch.object(multiindex_featureset.util, "error") as mock_error:
            result = multiindex_featureset._extract_embeddings_with_error_handling(
                extract_fn
            )
        mock_error.assert_not_called()
        assert len(result) == 4

    def test_custom_threshold_from_config(self, multiindex_featureset):
        """A custom fail_threshold from config is respected."""
        # Set threshold to 10%, then 2/5 = 40% should exceed it
        glob_conf.config["FEATS"]["fail_threshold"] = "0.1"
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={1, 3})
        with patch.object(multiindex_featureset.util, "error") as mock_error:
            mock_error.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                multiindex_featureset._extract_embeddings_with_error_handling(
                    extract_fn
                )
        assert mock_error.called

    def test_exact_threshold_no_abort(self, multiindex_featureset):
        """When failure rate exactly equals the threshold, no abort occurs (strict >)."""
        # 2 out of 5 fails = 40%, set threshold to exactly 0.4
        glob_conf.config["FEATS"]["fail_threshold"] = "0.4"
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={1, 3})
        with patch.object(multiindex_featureset.util, "error") as mock_error:
            result = multiindex_featureset._extract_embeddings_with_error_handling(
                extract_fn
            )
        mock_error.assert_not_called()
        assert len(result) == 3

    def test_invalid_threshold_falls_back_to_default(self, multiindex_featureset):
        """A non-numeric fail_threshold falls back to the default instead of crashing."""
        glob_conf.config["FEATS"]["fail_threshold"] = "not-a-number"
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={2})
        with patch.object(multiindex_featureset.util, "error") as mock_error:
            result = multiindex_featureset._extract_embeddings_with_error_handling(
                extract_fn
            )
        # 1/5 = 20%, below the default 50% threshold, so no abort
        mock_error.assert_not_called()
        assert len(result) == 4

    def test_out_of_range_threshold_is_clamped(self, multiindex_featureset):
        """A fail_threshold above 1.0 is clamped to 1.0, so even all failures don't abort."""
        glob_conf.config["FEATS"]["fail_threshold"] = "2.0"
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={0, 1, 2, 3, 4})
        with patch.object(multiindex_featureset.util, "error") as mock_error:
            multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        mock_error.assert_not_called()

    def test_assertion_error_counted_as_failure(self, multiindex_featureset):
        """AssertionError (e.g. sample-rate mismatch) is treated as a per-file failure."""
        call_count = {"n": 0}

        def extract_fn(file, start, end):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx == 2:
                assert False, "got 8000 instead of 16000"
            return np.ones(4)

        with patch.object(multiindex_featureset.util, "error") as mock_error:
            result = multiindex_featureset._extract_embeddings_with_error_handling(
                extract_fn
            )
        mock_error.assert_not_called()
        assert len(result) == 4

    def test_keyboard_interrupt_not_caught(self, multiindex_featureset):
        """KeyboardInterrupt must propagate and not be swallowed."""
        call_count = {"n": 0}

        def extract_fn(file, start, end):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx == 2:
                raise KeyboardInterrupt()
            return np.ones(4)

        with pytest.raises(KeyboardInterrupt):
            multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)

    def test_failure_summary_contains_count(self, multiindex_featureset):
        """Summary warning should contain failure count and percentage."""
        extract_fn = self._make_extract_fn(emb_dim=4, fail_indices={0, 4})
        with patch.object(multiindex_featureset.util, "warn") as mock_warn:
            multiindex_featureset._extract_embeddings_with_error_handling(extract_fn)
        summary_calls = [
            str(call)
            for call in mock_warn.call_args_list
            if "2/5" in str(call) and "40.0%" in str(call)
        ]
        assert len(summary_calls) == 1

    def test_keyboard_interrupt_not_swallowed(self, multiindex_featureset):
        """KeyboardInterrupt must propagate through the extraction loop."""

        def extract_fn_interrupt(file, start, end):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            multiindex_featureset._extract_embeddings_with_error_handling(
                extract_fn_interrupt
            )

    def test_type_error_not_swallowed(self, multiindex_featureset):
        """TypeError (programming error) must propagate, not be silently swallowed."""

        def extract_fn_type_error(file, start, end):
            raise TypeError("unexpected argument")

        with pytest.raises(TypeError):
            multiindex_featureset._extract_embeddings_with_error_handling(
                extract_fn_type_error
            )

    def test_embedding_values_correct(self, multiindex_featureset, multiindex_data_df):
        """Embeddings returned by extract_fn appear as rows in the result DataFrame."""
        rng = np.random.default_rng(0)
        embeddings = [rng.random(6) for _ in range(len(multiindex_data_df))]
        call_count = {"n": 0}

        def extract_fn(file, start, end):
            emb = embeddings[call_count["n"]]
            call_count["n"] += 1
            return emb

        result = multiindex_featureset._extract_embeddings_with_error_handling(
            extract_fn
        )
        assert result.shape == (5, 6)
        for i in range(5):
            np.testing.assert_array_almost_equal(result.iloc[i].values, embeddings[i])
