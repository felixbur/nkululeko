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
