"""Unit tests for AugmenterAuglib (nkululeko/augmenting/augmenter_auglib.py).

audb.load()/audb.load_media() download real external assets over the
network, so they are mocked throughout -- these tests exercise which assets
get downloaded for a given AUGMENT.transformations selection, which is
AugmenterAuglib's own responsibility (and was previously buggy: every asset
was downloaded unconditionally, regardless of what was actually selected).
"""

import configparser
from unittest.mock import MagicMock, patch

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
    import pandas as pd

    idx = pd.MultiIndex.from_arrays([[], [], []], names=["file", "start", "end"])
    return pd.DataFrame({"label": []}, index=idx)


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
