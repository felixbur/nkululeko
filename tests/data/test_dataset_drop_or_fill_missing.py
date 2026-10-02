"""Tests for nkululeko/data/dataset.py — Dataset._drop_or_fill_missing().

Regression for GH #455: Dataset.load() used a blanket dropna(), so any row
missing an auto-probed demographic column (age/gender/speaker) -- not just
a missing target -- was silently dropped. A source database lacking a
demographic scheme entirely therefore lost every one of its rows, with no
row count logged and no warning. Only the target should require a
non-null value; other columns get the literal string "na" instead of
dropping the row, and both the drop and the fill are logged.
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

    def test_missing_demographic_values_filled_with_na_string(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame(
            {"emotion": ["happy", "sad"], "gender": ["male", None]}
        )
        out = ds._drop_or_fill_missing(df)
        assert out["gender"].tolist() == ["male", "na"]

    def test_drop_and_fill_are_logged(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame(
            {"emotion": ["happy", None, "sad"], "gender": ["male", "female", None]}
        )
        ds._drop_or_fill_missing(df)
        debug_calls = [c.args[0] for c in ds.util.debug.call_args_list]
        assert any("dropped" in m and "missing target" in m for m in debug_calls)
        assert any("filled missing values" in m for m in debug_calls)

    def test_no_missing_values_is_a_noop(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame({"emotion": ["happy", "sad"], "gender": ["male", "female"]})
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 2
        assert out["gender"].tolist() == ["male", "female"]

    def test_col_label_none_falls_back_to_blanket_dropna(self):
        """No target resolved at all (e.g. no columns, no label) -- df is
        just an index-only frame, so the old blanket dropna() behavior is
        preserved rather than crashing on subset=[None]."""
        ds = _make_dataset(None)
        df = pd.DataFrame(index=["f1.wav", "f2.wav"])
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 2

    def test_col_label_not_in_df_falls_back_to_blanket_dropna(self):
        ds = _make_dataset("emotion")
        df = pd.DataFrame({"gender": ["male", None]})
        out = ds._drop_or_fill_missing(df)
        assert out.shape[0] == 1
