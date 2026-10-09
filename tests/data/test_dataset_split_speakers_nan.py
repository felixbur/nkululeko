"""Tests for nkululeko/data/dataset.py — Dataset.split_speakers()/split_speakers_3().

Regression for GH #461: rows with no speaker id (now kept by
Dataset._drop_or_fill_missing(), GH #455) must not be treated as one
single speaker when splitting by speaker. pandas' Series.unique() includes
NaN as a candidate value and Series.isin([nan]) matches every NaN row, so
sampling directly from df.speaker.unique() could draw NaN as a "speaker"
and dump every row with no speaker id into a single split, or split a
genuinely speaker-disjoint database inconsistently. Rows with no speaker
id are now excluded from the sampling population and always kept in
train.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from nkululeko.data.dataset import Dataset


def _make_segmented_index(files):
    arrays = [
        files,
        [timedelta(0)] * len(files),
        [timedelta(seconds=1)] * len(files),
    ]
    return pd.MultiIndex.from_arrays(arrays, names=["file", "start", "end"])


def _make_dataset(df, test_size=100, dev_size=0):
    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.df = df
    util = MagicMock()
    sizes = {"test_size": test_size, "dev_size": dev_size}
    util.config_val_data.side_effect = (
        lambda name, key, default: sizes.get(key, default)
    )
    ds.util = util
    return ds


# Deterministic stand-in for random.sample: picks from the *end* of
# whatever population it's given, so if NaN is still in the population
# (the pre-fix bug) it gets selected, whereas the fix excludes NaN from
# the population before sampling ever happens.
def _sample_from_end(population, k):
    return list(population)[-k:] if k else []


class TestSplitSpeakersExcludesNanFromSampling:
    def test_rows_with_no_speaker_are_kept_in_train(self, monkeypatch):
        files = [f"f{i}.wav" for i in range(6)]
        idx = _make_segmented_index(files)
        df = pd.DataFrame(
            {
                "speaker": ["s1", "s1", "s2", "s2", np.nan, np.nan],
                "emotion": ["happy"] * 6,
            },
            index=idx,
        )
        ds = _make_dataset(df, test_size=100)
        monkeypatch.setattr("nkululeko.data.dataset.sample", _sample_from_end)

        ds.split_speakers()

        assert ds.df_test["speaker"].notna().all()
        assert ds.df_train["speaker"].isna().sum() == 2

    def test_split_speakers_3_also_excludes_nan_from_sampling(self, monkeypatch):
        files = [f"f{i}.wav" for i in range(8)]
        idx = _make_segmented_index(files)
        df = pd.DataFrame(
            {
                "speaker": ["s1", "s1", "s2", "s2", "s3", "s3", np.nan, np.nan],
                "emotion": ["happy"] * 8,
            },
            index=idx,
        )
        ds = _make_dataset(df, test_size=100, dev_size=0)
        monkeypatch.setattr("nkululeko.data.dataset.sample", _sample_from_end)

        ds.split_speakers_3()

        assert ds.df_test["speaker"].notna().all()
        assert ds.df_dev["speaker"].notna().all()
        assert ds.df_train["speaker"].isna().sum() == 2
