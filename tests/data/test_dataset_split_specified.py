"""Tests for nkululeko/data/dataset.py — Dataset.split(), "specified" strategy.

Regression for GH #464 (follow-up to GH #455/ed188c5): split()'s "specified"
branch (DATA.<name>.test_tables/train_tables) still applied a blanket
dropna() to the concatenated split tables, dropping rows missing a value in
ANY column of that table -- even one that's never used (only the target is
copied back onto self.df_test/self.df_train, and the table is otherwise
only used for its index). ed188c5 already fixed the equivalent problem in
Dataset.load() by requiring only the target to be non-null; this sibling
code path never got that fix.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd

from nkululeko.data.dataset import Dataset


def _make_segmented_index(files):
    arrays = [
        files,
        [timedelta(0)] * len(files),
        [timedelta(seconds=1)] * len(files),
    ]
    return pd.MultiIndex.from_arrays(arrays, names=["file", "start", "end"])


def _make_dataset(raw_df, config_values):
    files = ["f1.wav", "f2.wav"]
    index = _make_segmented_index(files)

    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.target = "emotion"
    ds.df = pd.DataFrame({"emotion": ["happy", "sad"]}, index=index)
    ds.is_labeled = True
    ds.got_speaker = False
    ds.split3 = False
    ds.db = type(
        "FakeDb", (), {"tables": {"raw_table": type("T", (), {"df": raw_df})()}}
    )()

    util = MagicMock()
    util.config_val_data.side_effect = (
        lambda name, key, default: config_values.get(key, default)
    )
    ds.util = util
    return ds


class TestSplitSpecifiedDoesNotDropOnUnusedColumnNan:
    def test_test_tables_keeps_rows_missing_an_unused_column(self):
        index = _make_segmented_index(["f1.wav", "f2.wav"])
        raw_df = pd.DataFrame(
            {"emotion": ["happy", "sad"], "predicted": [None, "sad"]}, index=index
        )
        ds = _make_dataset(
            raw_df, {"split_strategy": "specified", "test_tables": "['raw_table']"}
        )
        ds.split()
        assert len(ds.df_test) == 2
        assert list(ds.df_test["emotion"]) == ["happy", "sad"]

    def test_train_tables_keeps_rows_missing_an_unused_column(self):
        index = _make_segmented_index(["f1.wav", "f2.wav"])
        raw_df = pd.DataFrame(
            {"emotion": ["happy", "sad"], "predicted": [None, "sad"]}, index=index
        )
        ds = _make_dataset(
            raw_df, {"split_strategy": "specified", "train_tables": "['raw_table']"}
        )
        ds.split()
        assert len(ds.df_train) == 2
        assert list(ds.df_train["emotion"]) == ["happy", "sad"]

    def test_row_missing_the_target_itself_is_still_dropped(self):
        index = _make_segmented_index(["f1.wav", "f2.wav"])
        raw_df = pd.DataFrame({"emotion": [None, "sad"]}, index=index)
        ds = _make_dataset(
            raw_df, {"split_strategy": "specified", "test_tables": "['raw_table']"}
        )
        ds.split()
        assert len(ds.df_test) == 1
        assert list(ds.df_test["emotion"]) == ["sad"]
