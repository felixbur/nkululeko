"""Unit tests for AugmenterAuglib (nkululeko/augmenting/augmenter_auglib.py).

audb.load()/audb.load_media() download real external assets over the
network, so they are mocked throughout -- these tests exercise which assets
get downloaded for a given AUGMENT.transformations selection, which is
AugmenterAuglib's own responsibility (and was previously buggy: every asset
was downloaded unconditionally, regardless of what was actually selected).
"""

import configparser
from unittest.mock import MagicMock, patch

import audiofile
import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.augmenting.augmenter_auglib import AugmenterAuglib


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    config["AUGMENT"] = {}
    glob_conf.config = config
    yield
    glob_conf.config = None


def _fake_db(*files):
    db = MagicMock()
    db.files = list(files)
    return db


def _make_augmenter(transformations, audb_load, audb_load_media=None):
    glob_conf.config["AUGMENT"]["transformations"] = str(transformations)
    with patch(
        "nkululeko.augmenting.augmenter_auglib.audb.load", side_effect=audb_load
    ) as mock_load, patch(
        "nkululeko.augmenting.augmenter_auglib.audb.load_media",
        side_effect=audb_load_media or (lambda *a, **k: []),
    ) as mock_load_media:
        augmenter = AugmenterAuglib(_empty_df())
    return augmenter, mock_load, mock_load_media


def _empty_df():
    idx = pd.MultiIndex.from_arrays([[], [], []], names=["file", "start", "end"])
    return pd.DataFrame({"label": []}, index=idx)


def _make_wav(path, sr=16000, duration=1.0, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    audiofile.write(str(path), signal=signal, sampling_rate=sr)


def _make_df(files):
    starts = [pd.Timedelta(0)] * len(files)
    ends = [pd.Timedelta(seconds=1)] * len(files)
    idx = pd.MultiIndex.from_arrays(
        [files, starts, ends], names=["file", "start", "end"]
    )
    return pd.DataFrame({"label": range(len(files))}, index=idx)


def _make_segmented_df(files, starts, ends):
    idx = pd.MultiIndex.from_arrays(
        [files, starts, ends], names=["file", "start", "end"]
    )
    return pd.DataFrame({"label": range(len(files))}, index=idx)


def _build_augmenter(df):
    """Build a real AugmenterAuglib with no transforms selected -- avoids
    any audb download while still exercising the real augment() file loop
    (NormalizeByPeak is always applied, so self.augmenter stays callable)."""
    glob_conf.config["AUGMENT"]["transformations"] = "[]"
    with patch("nkululeko.augmenting.augmenter_auglib.audb.load") as mock_load, patch(
        "nkululeko.augmenting.augmenter_auglib.audb.load_media"
    ) as mock_load_media:
        augmenter = AugmenterAuglib(df)
    mock_load.assert_not_called()
    mock_load_media.assert_not_called()
    return augmenter


class TestAugmenterAuglibAssetLoading:
    def test_noise_only_downloads_nothing(self):
        """Pink noise needs no external asset at all."""
        _, mock_load, mock_load_media = _make_augmenter(
            ["noise"], audb_load=lambda *a, **k: _fake_db("x.wav")
        )
        mock_load.assert_not_called()
        mock_load_media.assert_not_called()

    def test_basics_and_room_download_only_room_and_speech(self):
        """['noise', 'babble', 'room'] -- the basics + RIR -- must not
        trigger the 'music' table or the cough media download."""
        _, mock_load, mock_load_media = _make_augmenter(
            ["noise", "babble", "room"],
            audb_load=lambda *a, **k: _fake_db("x.wav"),
        )
        mock_load_media.assert_not_called()
        called_tables = [
            call.kwargs.get("tables") for call in mock_load.call_args_list
        ]
        assert "rir" in called_tables
        assert "speech" in called_tables
        assert "music" not in called_tables

    def test_music_only_downloads_music_not_speech_or_room(self):
        _, mock_load, mock_load_media = _make_augmenter(
            ["music"], audb_load=lambda *a, **k: _fake_db("x.wav")
        )
        mock_load_media.assert_not_called()
        called_tables = [
            call.kwargs.get("tables") for call in mock_load.call_args_list
        ]
        assert called_tables == ["music"]

    def test_cough_downloads_only_cough_media(self):
        _, mock_load, mock_load_media = _make_augmenter(
            ["cough"],
            audb_load=lambda *a, **k: _fake_db("x.wav"),
            audb_load_media=lambda *a, **k: ["cough1.wav"],
        )
        mock_load.assert_not_called()
        mock_load_media.assert_called_once()

    def test_all_default_transforms_download_every_asset(self):
        _, mock_load, mock_load_media = _make_augmenter(
            ["room", "music", "noise", "babble", "crop", "cough"],
            audb_load=lambda *a, **k: _fake_db("x.wav"),
            audb_load_media=lambda *a, **k: ["cough1.wav"],
        )
        called_tables = [
            call.kwargs.get("tables") for call in mock_load.call_args_list
        ]
        assert sorted(called_tables) == ["music", "rir", "speech"]
        mock_load_media.assert_called_once()


class TestAugmenterAuglibAugment:
    """Regression: augment() iterated every *row's* file index value, so a
    segmented dataframe (multiple (start, end) rows per file) redundantly
    re-ran the randomized augmentation pipeline once per segment, with
    whichever run happened last mapped to every segment of that file. Also,
    the output path was built from a hardcoded '/' split rather than
    os.path, which isn't portable to Windows paths."""

    def test_same_file_augmented_once_across_multiple_segments(self, tmp_path):
        wav_dir = tmp_path / "audio"
        wav_dir.mkdir()
        wav_path = wav_dir / "f0.wav"
        _make_wav(wav_path)
        df = _make_segmented_df(
            [str(wav_path)] * 3,
            [pd.Timedelta(0), pd.Timedelta(seconds=1), pd.Timedelta(seconds=2)],
            [
                pd.Timedelta(seconds=1),
                pd.Timedelta(seconds=2),
                pd.Timedelta(seconds=3),
            ],
        )
        augmenter = _build_augmenter(df)
        call_count = {"n": 0}
        original_call = augmenter.augmenter

        def counting_call(signal, sr):
            call_count["n"] += 1
            return original_call(signal, sr)

        augmenter.augmenter = counting_call

        df_ret = augmenter.augment("all")

        assert call_count["n"] == 1
        assert len(df_ret) == 3
        assert len(set(df_ret.index.get_level_values(0))) == 1

    def test_same_subfolder_and_filename_in_different_datasets_dont_collide(
        self, tmp_path
    ):
        dataset_a = tmp_path / "dataset_a" / "wav"
        dataset_b = tmp_path / "dataset_b" / "wav"
        dataset_a.mkdir(parents=True)
        dataset_b.mkdir(parents=True)
        path_a = dataset_a / "f0.wav"
        path_b = dataset_b / "f0.wav"
        _make_wav(path_a, freq=200.0)
        _make_wav(path_b, freq=800.0)

        augmenter = _build_augmenter(_make_df([str(path_a), str(path_b)]))
        df_ret = augmenter.augment("all")

        new_files = list(df_ret.index.get_level_values(0))
        assert len(set(new_files)) == 2
        signal_a, _ = audiofile.read(new_files[0])
        signal_b, _ = audiofile.read(new_files[1])
        assert not np.allclose(signal_a, signal_b)
