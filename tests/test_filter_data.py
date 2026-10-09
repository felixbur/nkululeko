"""Tests for nkululeko/filter_data.py — DataFilter class."""

import configparser
from datetime import timedelta

import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.filter_data import DataFilter


def _make_segmented_index(entries):
    """Build audformat-style MultiIndex from (file, start_s, end_s) tuples."""
    files = [e[0] for e in entries]
    starts = [timedelta(seconds=e[1]) for e in entries]
    ends = [timedelta(seconds=e[2]) for e in entries]
    return pd.MultiIndex.from_arrays(
        [files, starts, ends], names=["file", "start", "end"]
    )


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "test", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['test_db']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.init_config(config)
    yield
    glob_conf.config = None


@pytest.fixture
def simple_df():
    """Plain DataFrame with emotion column and is_labeled flag."""
    df = pd.DataFrame({"emotion": ["happy", "sad", "angry", "happy", "sad"]})
    df.is_labeled = True
    df.got_gender = False
    df.got_speaker = False
    return df


@pytest.fixture
def speaker_df():
    """DataFrame with speaker column for speaker-limit tests."""
    df = pd.DataFrame(
        {
            "emotion": ["happy", "sad", "angry", "happy", "sad", "angry"],
            "speaker": ["s1", "s1", "s1", "s2", "s2", "s2"],
        }
    )
    df.is_labeled = True
    df.got_gender = False
    df.got_speaker = True
    return df


@pytest.fixture
def segmented_df():
    """DataFrame with audformat segmented MultiIndex for duration tests."""
    entries = [
        ("a.wav", 0.0, 0.5),  # 0.5 s — short
        ("b.wav", 0.0, 2.0),  # 2.0 s — medium
        ("c.wav", 0.0, 5.0),  # 5.0 s — long
        ("d.wav", 0.0, 10.0),  # 10.0 s — very long
    ]
    idx = _make_segmented_index(entries)
    df = pd.DataFrame({"emotion": ["happy", "sad", "angry", "happy"]}, index=idx)
    df.is_labeled = True
    df.got_gender = False
    df.got_speaker = False
    return df


class TestDataFilterInit:
    def test_df_is_copied(self, simple_df):
        f = DataFilter(simple_df)
        assert f.df is not simple_df
        assert f.df.equals(simple_df)

    def test_flags_copied(self, simple_df):
        f = DataFilter(simple_df)
        assert f.df.is_labeled is True


class TestLimitSamples:
    def test_no_limit_returns_unchanged(self, simple_df):
        f = DataFilter(simple_df)
        result = f.limit_samples()
        assert len(result) == len(simple_df)

    def test_limit_larger_than_df_returns_unchanged(self, simple_df, tmp_path):
        glob_conf.config["DATA"]["limit_samples"] = "100"
        f = DataFilter(simple_df)
        result = f.limit_samples()
        assert len(result) == len(simple_df)

    def test_limit_reduces_samples(self, simple_df):
        glob_conf.config["DATA"]["limit_samples"] = "3"
        f = DataFilter(simple_df)
        result = f.limit_samples()
        assert len(result) == 3

    def test_limit_is_random_subset(self, simple_df):
        glob_conf.config["DATA"]["limit_samples"] = "2"
        f = DataFilter(simple_df)
        result = f.limit_samples()
        assert set(result["emotion"]).issubset({"happy", "sad", "angry"})


class TestLimitSpeakers:
    def test_no_limit_returns_unchanged(self, speaker_df):
        f = DataFilter(speaker_df)
        result = f.limit_speakers()
        assert len(result) == len(speaker_df)

    def test_limit_per_speaker(self, speaker_df):
        glob_conf.config["DATA"]["limit_samples_per_speaker"] = "2"
        f = DataFilter(speaker_df)
        result = f.limit_speakers()
        # 2 speakers × max 2 samples each = 4
        assert len(result) == 4

    def test_limit_larger_than_speaker_samples_keeps_all(self, speaker_df):
        glob_conf.config["DATA"]["limit_samples_per_speaker"] = "10"
        f = DataFilter(speaker_df)
        result = f.limit_speakers()
        assert len(result) == len(speaker_df)

    def test_rows_with_no_speaker_id_are_kept_unfiltered(self):
        """Regression for GH #461: Series.eq(nan) never matches, so every
        per-speaker group for the NaN "speaker" came back empty and those
        rows were silently dropped entirely, instead of just not being
        subject to a per-speaker cap."""
        df = pd.DataFrame(
            {
                "emotion": ["happy", "sad", "angry", "happy", "sad"],
                "speaker": ["s1", "s1", "s1", None, None],
            }
        )
        df.is_labeled = True
        df.got_gender = False
        df.got_speaker = True
        glob_conf.config["DATA"]["limit_samples_per_speaker"] = "1"
        f = DataFilter(df)
        result = f.limit_speakers()
        # s1 capped to 1 sample; the 2 no-speaker rows pass through untouched.
        assert len(result) == 3
        assert result["speaker"].isna().sum() == 2


class TestFilterValue:
    def test_no_filter_returns_unchanged(self, simple_df):
        f = DataFilter(simple_df)
        result = f.filter_value()
        assert len(result) == len(simple_df)

    def test_list_filter_single_value(self, simple_df):
        glob_conf.config["DATA"]["filter"] = "[['emotion', 'happy']]"
        f = DataFilter(simple_df)
        result = f.filter_value()
        assert all(result["emotion"] == "happy")

    def test_list_filter_multiple_values(self, simple_df):
        glob_conf.config["DATA"]["filter"] = "[['emotion', ['happy', 'sad']]]"
        f = DataFilter(simple_df)
        result = f.filter_value()
        assert set(result["emotion"]) == {"happy", "sad"}

    def test_dict_filter_single_value(self, simple_df):
        glob_conf.config["DATA"]["filter"] = "{'emotion': 'angry'}"
        f = DataFilter(simple_df)
        result = f.filter_value()
        assert all(result["emotion"] == "angry")

    def test_dict_filter_list_value(self, simple_df):
        glob_conf.config["DATA"]["filter"] = "{'emotion': ['happy', 'angry']}"
        f = DataFilter(simple_df)
        result = f.filter_value()
        assert set(result["emotion"]) == {"happy", "angry"}

    def test_filter_removes_all_gives_empty(self, simple_df):
        glob_conf.config["DATA"]["filter"] = "[['emotion', 'nonexistent']]"
        f = DataFilter(simple_df)
        result = f.filter_value()
        assert len(result) == 0


class TestFilterDuration:
    def test_no_duration_config_returns_unchanged(self, segmented_df):
        f = DataFilter(segmented_df)
        result = f.filter_duration()
        assert len(result) == len(segmented_df)

    def test_min_duration_removes_short(self, segmented_df):
        glob_conf.config["DATA"]["min_duration_of_sample"] = "1.0"
        f = DataFilter(segmented_df)
        result = f.filter_duration()
        # 0.5 s sample is removed
        assert len(result) == 3

    def test_max_duration_removes_long(self, segmented_df):
        glob_conf.config["DATA"]["max_duration_of_sample"] = "5.0"
        f = DataFilter(segmented_df)
        result = f.filter_duration()
        # 10.0 s sample is removed
        assert len(result) == 3

    def test_min_and_max_duration_combined(self, segmented_df):
        glob_conf.config["DATA"]["min_duration_of_sample"] = "1.0"
        glob_conf.config["DATA"]["max_duration_of_sample"] = "5.0"
        f = DataFilter(segmented_df)
        result = f.filter_duration()
        # both filters applied sequentially: 0.5 s (< 1.0) and 10.0 s (> 5.0) removed
        assert len(result) == 2


class TestAllFilters:
    def test_all_filters_no_config_passthrough(self, simple_df):
        f = DataFilter(simple_df)
        result = f.all_filters()
        assert len(result) == len(simple_df)

    def test_all_filters_with_limit(self, simple_df):
        glob_conf.config["DATA"]["limit_samples"] = "2"
        f = DataFilter(simple_df)
        result = f.all_filters()
        assert len(result) == 2
