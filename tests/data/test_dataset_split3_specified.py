"""Tests for nkululeko/data/dataset.py — Dataset.split_3(), "specified" strategy.

Regression: split_strategy="specified" combined with DATA.<name>.colnames
(renaming the raw audformat target column to a friendlier name, e.g.
grbas_category -> grade) crashed with `KeyError: 'grade'`.

Root cause: split_3()'s "specified" branch reads train/dev/test rows
straight from the raw audformat table (self.db.tables[...].df), which
never went through the colnames renaming self.df already has, then tries
to copy `testdf[self.target]`/`traindf[self.target]`/`devdf[self.target]`
back onto self.df_test/df_train/df_dev - self.target is the *renamed* name,
so it's not a column on the raw table. The older, non-3-way split() already
guards the equivalent copy with `try/except KeyError: pass`; split_3() was
missing that guard entirely.
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


def _make_dataset(raw_column, config_values):
    files = ["f1.wav", "f2.wav"]
    index = _make_segmented_index(files)

    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.target = "grade"
    ds.split3 = True
    # self.df already went through the renaming db.get(self.col_label, ...)
    # would have done in load() - it has the friendly "grade" column.
    ds.df = pd.DataFrame({"grade": ["mild", "normal"]}, index=index)
    # The raw audformat table, as split_3() reads it directly - still has
    # the original, unrenamed column name.
    raw_df = pd.DataFrame({raw_column: ["mild", "normal"]}, index=index)
    ds.db = type("FakeDb", (), {"tables": {"raw_table": type("T", (), {"df": raw_df})()}})()

    util = MagicMock()
    util.config_val_data.side_effect = (
        lambda name, key, default: config_values.get(key, default)
    )
    ds.util = util
    return ds


class TestSplit3SpecifiedWithRenamedTarget:
    def test_dev_tables_survives_colname_rename(self):
        ds = _make_dataset(
            "grbas_category",
            {"split_strategy": "specified", "dev_tables": "['raw_table']"},
        )
        ds.split_3()  # must not raise KeyError: 'grade'
        assert not ds.df_dev.empty
        assert list(ds.df_dev["grade"]) == ["mild", "normal"]

    def test_test_tables_survives_colname_rename(self):
        ds = _make_dataset(
            "grbas_category",
            {"split_strategy": "specified", "test_tables": "['raw_table']"},
        )
        ds.split_3()
        assert not ds.df_test.empty
        assert list(ds.df_test["grade"]) == ["mild", "normal"]

    def test_train_tables_survives_colname_rename(self):
        ds = _make_dataset(
            "grbas_category",
            {"split_strategy": "specified", "train_tables": "['raw_table']"},
        )
        ds.split_3()
        assert not ds.df_train.empty
        assert list(ds.df_train["grade"]) == ["mild", "normal"]

    def test_copies_target_when_column_names_already_match(self):
        # No renaming involved: raw table's column is already "grade", so
        # the copy-back should still succeed (not just silently no-op).
        ds = _make_dataset(
            "grade",
            {"split_strategy": "specified", "dev_tables": "['raw_table']"},
        )
        ds.split_3()
        assert list(ds.df_dev["grade"]) == ["mild", "normal"]
