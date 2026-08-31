"""Tests for nkululeko/data/dataset.py — Dataset._get_tables().

Regression: split_strategy="specified" with a dev_tables entry (used for
[EXP] traindevtest=True 3-way splits) silently produced an empty dev set
whenever the dev table wasn't *also* listed under test_tables/train_tables.
Root cause: _get_tables() built the table list used to load Dataset.load()'s
self.df from target_tables/files_tables/test_tables/train_tables only, never
dev_tables - so rows belonging only to the dev table never made it into
self.df, and split_3()'s later `self.df.loc[self.df.index.intersection(
devdf.index)]` intersected against nothing.
"""

from unittest.mock import MagicMock

from nkululeko.data.dataset import Dataset


def _make_dataset(config_values):
    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    util = MagicMock()
    util.config_val_data.side_effect = (
        lambda name, key, default: config_values.get(key, default)
    )
    ds.util = util
    return ds


class TestGetTables:
    def test_includes_dev_tables(self):
        ds = _make_dataset(
            {
                "train_tables": "['train_tbl']",
                "dev_tables": "['dev_tbl']",
                "test_tables": "['test_tbl']",
            }
        )
        tables = ds._get_tables()
        assert "dev_tbl" in tables
        assert "train_tbl" in tables
        assert "test_tbl" in tables

    def test_dev_tables_alone(self):
        ds = _make_dataset({"dev_tables": "['dev_tbl']"})
        assert ds._get_tables() == ["dev_tbl"]

    def test_dev_tables_optional(self):
        ds = _make_dataset({"train_tables": "['train_tbl']"})
        assert ds._get_tables() == ["train_tbl"]

    def test_no_tables_configured(self):
        ds = _make_dataset({})
        assert ds._get_tables() == []
