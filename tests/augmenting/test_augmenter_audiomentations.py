"""Unit tests for AugmenterAudiomentations (nkululeko/augmenting/augmenter_audiomentations.py).

Regression: augment() iterated every *row's* file index value, so a
segmented dataframe (multiple (start, end) rows per file) redundantly
re-ran the randomized augmentation pipeline once per segment, with
whichever run happened last mapped to every segment of that file. Also,
the output path was built from a hardcoded '/' split rather than os.path,
which isn't portable to Windows paths.
"""

import configparser

import audiofile
import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.augmenting.augmenter_audiomentations import AugmenterAudiomentations


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


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


class TestAugmenterAudiomentationsAugment:
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
        augmenter = AugmenterAudiomentations(df)
        call_count = {"n": 0}
        original_call = augmenter.audioment

        def counting_call(samples, sample_rate):
            call_count["n"] += 1
            return original_call(samples=samples, sample_rate=sample_rate)

        augmenter.audioment = counting_call

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

        augmenter = AugmenterAudiomentations(_make_df([str(path_a), str(path_b)]))
        df_ret = augmenter.augment("all")

        new_files = list(df_ret.index.get_level_values(0))
        assert len(set(new_files)) == 2
