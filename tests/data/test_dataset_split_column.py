"""Tests for nkululeko/data/dataset.py -- split_strategy="column" (issue #423).

Lets a single database be split into train/test (or train/dev/test) by the
values of an arbitrary column, rather than by speaker or a fixed random
percentage, e.g. splitting by recording location:

    db.split_strategy = column
    db.split_column = location
    db.train_vals = ['tokyo', 'berlin']
    db.test_vals = ['paris']
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nkululeko.data.dataset import Dataset
from nkululeko.utils.errors import NkululukoError


def _make_dataset(df, config_values):
    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.df = df

    util = MagicMock()
    util.config_val_data.side_effect = (
        lambda name, key, default: config_values.get(key, default)
    )

    def _error(message):
        raise NkululukoError(f"ERROR: test: {message}")

    util.error.side_effect = _error
    ds.util = util
    return ds


class TestSplitByColumn:
    def test_train_test_split(self):
        df = pd.DataFrame(
            {"location": ["tokyo", "berlin", "paris", "paris", "tokyo"]},
            index=[0, 1, 2, 3, 4],
        )
        ds = _make_dataset(
            df,
            {
                "split_column": "location",
                "train_vals": "['tokyo', 'berlin']",
                "test_vals": "['paris']",
            },
        )

        ds.split_by_column()

        assert sorted(ds.df_train.index) == [0, 1, 4]
        assert sorted(ds.df_test.index) == [2, 3]

    def test_values_matching_neither_list_are_excluded(self):
        df = pd.DataFrame(
            {"location": ["tokyo", "unknown_city"]}, index=[0, 1]
        )
        ds = _make_dataset(
            df,
            {"split_column": "location", "train_vals": "['tokyo']", "test_vals": "[]"},
        )

        ds.split_by_column()

        assert list(ds.df_train.index) == [0]
        assert ds.df_test.empty

    def test_missing_vals_key_defaults_to_empty(self):
        """test_vals not configured at all (only train_vals given) --
        df_test should just be empty, not an error."""
        df = pd.DataFrame({"location": ["tokyo", "paris"]}, index=[0, 1])
        ds = _make_dataset(df, {"split_column": "location", "train_vals": "['tokyo']"})

        ds.split_by_column()

        assert list(ds.df_train.index) == [0]
        assert ds.df_test.empty

    def test_accepts_values_spelling_alias(self):
        """train_values/test_values (not just train_vals/test_vals) must
        also work -- this is the naming a real exp_colsplit.ini used."""
        df = pd.DataFrame(
            {"gender": ["male", "female", "male", "female"]}, index=[0, 1, 2, 3]
        )
        ds = _make_dataset(
            df,
            {
                "split_column": "gender",
                "train_values": "['male']",
                "test_values": "['female']",
            },
        )

        ds.split_by_column()

        assert sorted(ds.df_train.index) == [0, 2]
        assert sorted(ds.df_test.index) == [1, 3]

    def test_vals_spelling_takes_precedence_over_values(self):
        df = pd.DataFrame({"location": ["tokyo", "paris"]}, index=[0, 1])
        ds = _make_dataset(
            df,
            {
                "split_column": "location",
                "train_vals": "['tokyo']",
                "train_values": "['paris']",
            },
        )

        result = ds._column_split_values("train")

        assert result == ["tokyo"]

    def test_overlapping_train_test_vals_raises(self):
        """Reviewer follow-up: a value listed in both train_vals and
        test_vals would otherwise put the same rows in both splits,
        leaking train/test data."""
        df = pd.DataFrame(
            {"location": ["tokyo", "paris"]}, index=[0, 1]
        )
        ds = _make_dataset(
            df,
            {
                "split_column": "location",
                "train_vals": "['tokyo', 'paris']",
                "test_vals": "['paris']",
            },
        )

        with pytest.raises(NkululukoError, match="paris"):
            ds.split_by_column()

    def test_missing_split_column_config_raises(self):
        df = pd.DataFrame({"location": ["tokyo"]})
        ds = _make_dataset(df, {"train_vals": "['tokyo']"})

        with pytest.raises(NkululukoError, match="split_column"):
            ds.split_by_column()

    def test_unknown_split_column_raises(self):
        df = pd.DataFrame({"location": ["tokyo"]})
        ds = _make_dataset(
            df,
            {"split_column": "site", "train_vals": "['tokyo']", "test_vals": "[]"},
        )

        with pytest.raises(NkululukoError, match="site"):
            ds.split_by_column()


class TestSplitByColumn3:
    def test_train_dev_test_split(self):
        df = pd.DataFrame(
            {"location": ["tokyo", "berlin", "paris", "london"]},
            index=[0, 1, 2, 3],
        )
        ds = _make_dataset(
            df,
            {
                "split_column": "location",
                "train_vals": "['tokyo']",
                "dev_vals": "['berlin']",
                "test_vals": "['paris']",
            },
        )

        ds.split_by_column_3()

        assert list(ds.df_train.index) == [0]
        assert list(ds.df_dev.index) == [1]
        assert list(ds.df_test.index) == [2]
        # "london" (index 3) matches none of the configured value lists --
        # it's simply excluded from every split, not an error.

    def test_overlapping_dev_test_vals_raises(self):
        df = pd.DataFrame({"location": ["tokyo", "berlin"]}, index=[0, 1])
        ds = _make_dataset(
            df,
            {
                "split_column": "location",
                "train_vals": "['tokyo']",
                "dev_vals": "['berlin']",
                "test_vals": "['berlin']",
            },
        )

        with pytest.raises(NkululukoError, match="berlin"):
            ds.split_by_column_3()


class TestSplitDispatchesToColumnStrategy:
    """Dataset.split()/split_3() must actually route split_strategy="column"
    to the new methods, not just have them exist in isolation."""

    def test_split_routes_to_split_by_column(self):
        df = pd.DataFrame({"location": ["tokyo", "paris"]}, index=[0, 1])
        ds = _make_dataset(
            df,
            {
                "split_strategy": "column",
                "split_column": "location",
                "train_vals": "['tokyo']",
                "test_vals": "['paris']",
            },
        )
        ds.split3 = False
        ds.is_labeled = True

        ds.split()

        assert list(ds.df_train.index) == [0]
        assert list(ds.df_test.index) == [1]

    def test_split_3_routes_to_split_by_column_3(self):
        df = pd.DataFrame(
            {"location": ["tokyo", "berlin", "paris"]}, index=[0, 1, 2]
        )
        ds = _make_dataset(
            df,
            {
                "split_strategy": "column",
                "split_column": "location",
                "train_vals": "['tokyo']",
                "dev_vals": "['berlin']",
                "test_vals": "['paris']",
            },
        )
        ds.split3 = True

        ds.split_3()

        assert list(ds.df_train.index) == [0]
        assert list(ds.df_dev.index) == [1]
        assert list(ds.df_test.index) == [2]
