"""Tests for nkululeko/data/dataset.py — Dataset.prepare().

Regression: when a cached train/dev/test split is about to be reused (see
Datasplitter.fill_train_and_tests / should_reuse_split), Dataset.prepare()
must skip its own per-dataset filtering (DATA.required and DataFilter,
notably DATA.limit_samples/limit_speakers, which pick a fresh *random*
subsample every call). Otherwise self.df ends up reflecting a different
random sample than the one the cached split was actually computed from,
causing feats/labels misalignment later in Datasplitter.extract_feats() --
observed in the wild as "train feats (N) != train labels (M)" warnings
followed by an IndexError deep in model training.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nkululeko.data.dataset import Dataset


def _make_segmented_index(files):
    arrays = [
        files,
        [timedelta(0)] * len(files),
        [timedelta(seconds=1)] * len(files),
    ]
    return pd.MultiIndex.from_arrays(arrays, names=["file", "start", "end"])


def _make_dataset(df, split3=False):
    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.target = "emotion"
    ds.df = df
    ds.got_speaker = False
    ds.context = type("Ctx", (), {"split3": split3})()
    return ds


def _make_util():
    util = MagicMock()
    util.config_val_data.side_effect = lambda name, key, default: default
    util.exp_is_classification.return_value = True
    util.make_segmented_index.side_effect = lambda d: d
    return util


class TestPrepareSkipsFiltersOnSplitReuse:
    def test_skips_required_and_datafilter_when_reusing(self, monkeypatch):
        idx = _make_segmented_index(["/data/a.wav", "/data/b.wav"])
        df = pd.DataFrame({"emotion": ["happy", "sad"]}, index=idx)
        ds = _make_dataset(df)
        ds.util = _make_util()

        monkeypatch.setattr(
            "nkululeko.data.dataset.should_reuse_split", lambda util, split3: True
        )
        mock_filter_cls = MagicMock()
        monkeypatch.setattr("nkululeko.data.dataset.DataFilter", mock_filter_cls)

        ds.prepare()

        mock_filter_cls.assert_not_called()
        assert len(ds.df) == 2

    def test_applies_required_and_datafilter_when_not_reusing(self, monkeypatch):
        idx = _make_segmented_index(["/data/a.wav", "/data/b.wav"])
        df = pd.DataFrame({"emotion": ["happy", "sad"]}, index=idx)
        ds = _make_dataset(df)
        ds.util = _make_util()

        monkeypatch.setattr(
            "nkululeko.data.dataset.should_reuse_split", lambda util, split3: False
        )
        mock_filter_instance = MagicMock()
        mock_filter_instance.all_filters.return_value = df
        mock_filter_cls = MagicMock(return_value=mock_filter_instance)
        monkeypatch.setattr("nkululeko.data.dataset.DataFilter", mock_filter_cls)

        ds.prepare()

        mock_filter_cls.assert_called_once()
        mock_filter_instance.all_filters.assert_called_once_with(data_name="mydb")

    def test_random_limit_samples_filter_never_applied_on_reuse(self, monkeypatch):
        """The exact failure mode reported: DATA.limit_samples_per_speaker
        picks a *different* random subset on every call. On a cache-hit
        run, self.df must stay the full, unfiltered set -- a strict
        superset of whatever the cached split actually contains -- rather
        than a fresh, differently-random subsample."""
        idx = _make_segmented_index([f"/data/f_{i}.wav" for i in range(50)])
        df = pd.DataFrame(
            {
                "emotion": ["happy", "sad"] * 25,
                "speaker": ["s1"] * 50,
            },
            index=idx,
        )
        ds = _make_dataset(df)
        ds.got_speaker = True
        util = _make_util()
        # Simulate the real config: limit_samples_per_speaker=8.
        util.config_val_data.side_effect = (
            lambda name, key, default: "8"
            if key == "limit_samples_per_speaker"
            else default
        )
        ds.util = util

        monkeypatch.setattr(
            "nkululeko.data.dataset.should_reuse_split", lambda util, split3: True
        )

        ds.prepare()

        # Not reduced to 8 -- the random per-dataset filter never ran.
        assert len(ds.df) == 50


class _PassthroughDataFilter:
    """Stand-in for DataFilter that leaves self.df untouched, so the
    required-column filter's effect (applied just before DataFilter is
    constructed) survives to the assertion."""

    def __init__(self, df, context=None):
        self.df = df

    def all_filters(self, data_name=None):
        return self.df


class TestPrepareRequiredFilterHandlesRealNan:
    """GH #461: an earlier version of Dataset._drop_or_fill_missing() (GH
    #456) filled missing non-target values with the literal string "na",
    which made DATA.<name>.required's filter conflate that sentinel with a
    column that legitimately contains the real value "na" as data (e.g. a
    language code). Reverted to real NaN (GH #461), so the required
    filter is a plain notna() again: real NaN is still filtered, and a
    genuine "na" string value is correctly kept."""

    def test_real_nan_rows_are_filtered(self, monkeypatch):
        idx = _make_segmented_index([f"/data/f_{i}.wav" for i in range(3)])
        df = pd.DataFrame(
            {
                "emotion": ["happy", "sad", "angry"],
                "gender": ["male", None, "female"],
            },
            index=idx,
        )
        ds = _make_dataset(df)
        util = _make_util()
        util.config_val_data.side_effect = (
            lambda name, key, default: "gender" if key == "required" else default
        )
        ds.util = util

        monkeypatch.setattr(
            "nkululeko.data.dataset.should_reuse_split", lambda util, split3: False
        )
        monkeypatch.setattr("nkululeko.data.dataset.DataFilter", _PassthroughDataFilter)

        ds.prepare()

        assert len(ds.df) == 2

    def test_genuine_na_string_value_is_not_treated_as_missing(self, monkeypatch):
        """A column whose real data legitimately includes the string "na"
        (e.g. a language code) must not be filtered out by `required`."""
        idx = _make_segmented_index([f"/data/f_{i}.wav" for i in range(3)])
        df = pd.DataFrame(
            {
                "emotion": ["happy", "sad", "angry"],
                "language": ["en", "na", "de"],
            },
            index=idx,
        )
        ds = _make_dataset(df)
        util = _make_util()
        util.config_val_data.side_effect = (
            lambda name, key, default: "language" if key == "required" else default
        )
        ds.util = util

        monkeypatch.setattr(
            "nkululeko.data.dataset.should_reuse_split", lambda util, split3: False
        )
        monkeypatch.setattr("nkululeko.data.dataset.DataFilter", _PassthroughDataFilter)

        ds.prepare()

        assert len(ds.df) == 3
        assert "na" in ds.df["language"].values
