"""Tests for nkululeko/data/dataset.py — Dataset._drop_or_fill_missing().

Regression for GH #455: Dataset.load() used a blanket dropna(), so any row
missing an auto-probed demographic column (age/gender/speaker) -- not just
a missing target -- was silently dropped. A source database lacking a
demographic scheme entirely therefore lost every one of its rows, with no
row count logged and no warning. Only the target should require a
non-null value; other columns keep their real NaN and the drop is logged.

An earlier version of this fix (GH #456) filled other columns with the
literal string "na" instead of leaving them NaN, to make missing data
visible. GH #461 found that this broke on Categorical-dtype scheme
columns (fillna raises if "na" isn't already a category), silently
converted numeric columns to object dtype, and made every missing
speaker collapse into one fake "na" speaker for speaker-based splitting
and limit_samples_per_speaker. Reverted to real NaN, which needs no
special-casing by any downstream consumer.
"""

from unittest.mock import MagicMock

import pandas as pd

from nkululeko.data.dataset import Dataset


def _make_dataset(col_label):
    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.col_label = col_label
    ds.util = MagicMock()
    return ds


class TestDropOrFillMissing:
    def test_row_with_missing_target_is_dropped(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame(
            {"emotion": ["happy", None, "sad"], "gender": ["male", "female", None]}
        )
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 2
        assert out["emotion"].isna().sum() == 0

    def test_row_missing_only_a_demographic_column_is_kept(self):
        """The core regression: a whole database lacking a demographic
        scheme (e.g. no gender at all) must not lose every row."""
        ds = _make_dataset("emotion")
        df = pd.DataFrame(
            {"emotion": ["happy", "sad", "angry"], "gender": [None, None, None]}
        )
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 3

    def test_missing_demographic_values_stay_real_nan(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame(
            {"emotion": ["happy", "sad"], "gender": ["male", None]}
        )
        out = ds._drop_or_fill_missing(df)
        assert out["gender"].iloc[0] == "male"
        assert pd.isna(out["gender"].iloc[1])

    def test_drop_and_remaining_nans_are_logged(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame(
            {"emotion": ["happy", None, "sad"], "gender": ["male", "female", None]}
        )
        ds._drop_or_fill_missing(df)
        debug_calls = [c.args[0] for c in ds.util.debug.call_args_list]
        assert any("dropped" in m and "missing target" in m for m in debug_calls)
        assert any("missing (NaN) values kept in" in m for m in debug_calls)

    def test_no_missing_values_is_a_noop(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame({"emotion": ["happy", "sad"], "gender": ["male", "female"]})
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 2
        assert out["gender"].tolist() == ["male", "female"]

    def test_col_label_none_keeps_all_rows(self):
        """No target resolved at all (e.g. no columns, no label) -- there's
        nothing to require non-null for, so nothing should be dropped
        either (not even a blanket dropna(), which previously
        reintroduced the #455 bug for unlabeled/predict-only loading --
        GH #461)."""
        ds = _make_dataset(None)
        df = pd.DataFrame(index=["f1.wav", "f2.wav"])
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 2

    def test_col_label_not_in_df_keeps_rows_with_missing_other_columns(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame({"gender": ["male", None]})
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 2
