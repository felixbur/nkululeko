"""Unit tests for AugmenterSilero (nkululeko/augmenting/augmenter_silero.py).

torch.hub.load downloads the actual Silero model over the network, so it is
mocked throughout -- these tests exercise the file I/O, sample-rate handling
and dataframe-reindexing logic that AugmenterSilero itself is responsible for.
"""

import configparser
from unittest.mock import MagicMock, patch

import audiofile
import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.augmenting.augmenter_silero import AugmenterSilero


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


def _make_wav(path, sr, duration=1.0, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    audiofile.write(str(path), signal=signal, sampling_rate=sr)
    return signal


def _make_df(files):
    starts = [pd.Timedelta(0)] * len(files)
    ends = [pd.Timedelta(seconds=1)] * len(files)
    idx = pd.MultiIndex.from_arrays(
        [files, starts, ends], names=["file", "start", "end"]
    )
    return pd.DataFrame({"label": range(len(files))}, index=idx)


def _fake_torch_hub_load_factory(denoise_fn):
    """Build a stand-in for torch.hub.load that returns (model, samples, utils)."""

    def _load(*args, **kwargs):
        utils = (MagicMock(), MagicMock(), denoise_fn)
        return MagicMock(), MagicMock(), utils
    return _load


class TestAugmenterSilero:
    def test_denoise_at_same_rate_needs_no_resample(self, tmp_path):
        """When the denoiser's output is already at the original sampling
        rate, the file is used as-is (no extra resample-back step)."""
        wav_dir = tmp_path / "audio"
        wav_dir.mkdir()
        wav_path = wav_dir / "f0.wav"
        _make_wav(wav_path, sr=16000)

        def denoise(model, input_path, output_path, device="cpu"):
            signal, sr = audiofile.read(input_path)
            audiofile.write(output_path, signal=signal, sampling_rate=sr)
            return signal, sr

        with patch(
            "torch.hub.load", side_effect=_fake_torch_hub_load_factory(denoise)
        ):
            augmenter = AugmenterSilero(_make_df([str(wav_path)]))
            df_ret = augmenter.augment("all")

        new_file = df_ret.index.get_level_values(0)[0]
        assert new_file != str(wav_path)
        assert audiofile.sampling_rate(new_file) == 16000

    def test_denoise_at_different_rate_is_resampled_back(self, tmp_path):
        """Silero's model may output at a different sampling rate than the
        input (e.g. 48kHz); the augmented file must end up back at the
        original rate so it stays consistent with the rest of the samples."""
        wav_dir = tmp_path / "audio"
        wav_dir.mkdir()
        wav_path = wav_dir / "f0.wav"
        _make_wav(wav_path, sr=16000)

        def denoise(model, input_path, output_path, device="cpu"):
            signal, _ = audiofile.read(input_path)
            # simulate the model producing audio at a different sample rate
            audiofile.write(output_path, signal=signal, sampling_rate=48000)
            return signal, 48000

        with patch(
            "torch.hub.load", side_effect=_fake_torch_hub_load_factory(denoise)
        ):
            augmenter = AugmenterSilero(_make_df([str(wav_path)]))
            df_ret = augmenter.augment("all")

        new_file = df_ret.index.get_level_values(0)[0]
        assert audiofile.sampling_rate(new_file) == 16000

    def test_index_start_end_preserved(self, tmp_path):
        wav_dir = tmp_path / "audio"
        wav_dir.mkdir()
        wav_path = wav_dir / "f0.wav"
        _make_wav(wav_path, sr=16000)
        df = _make_df([str(wav_path)])

        def denoise(model, input_path, output_path, device="cpu"):
            signal, sr = audiofile.read(input_path)
            audiofile.write(output_path, signal=signal, sampling_rate=sr)
            return signal, sr

        with patch(
            "torch.hub.load", side_effect=_fake_torch_hub_load_factory(denoise)
        ):
            augmenter = AugmenterSilero(df)
            df_ret = augmenter.augment("all")

        assert df_ret.index.get_level_values(1)[0] == df.index.get_level_values(1)[0]
        assert df_ret.index.get_level_values(2)[0] == df.index.get_level_values(2)[0]
        assert list(df_ret["label"]) == list(df["label"])

    def test_model_loaded_with_configured_variant(self, tmp_path):
        glob_conf.config["AUGMENT"] = {"silero_model": "large_fast"}
        wav_dir = tmp_path / "audio"
        wav_dir.mkdir()
        wav_path = wav_dir / "f0.wav"
        _make_wav(wav_path, sr=16000)

        def denoise(model, input_path, output_path, device="cpu"):
            signal, sr = audiofile.read(input_path)
            audiofile.write(output_path, signal=signal, sampling_rate=sr)
            return signal, sr

        with patch(
            "torch.hub.load", side_effect=_fake_torch_hub_load_factory(denoise)
        ) as mock_load:
            AugmenterSilero(_make_df([str(wav_path)]))

        _, kwargs = mock_load.call_args
        assert kwargs["model"] == "silero_denoise"
        assert kwargs["name"] == "large_fast"

    def test_multiple_files_each_get_own_output(self, tmp_path):
        wav_dir = tmp_path / "audio"
        wav_dir.mkdir()
        paths = []
        for i in range(3):
            p = wav_dir / f"f{i}.wav"
            _make_wav(p, sr=16000, freq=200 + i * 100)
            paths.append(str(p))

        def denoise(model, input_path, output_path, device="cpu"):
            signal, sr = audiofile.read(input_path)
            audiofile.write(output_path, signal=signal, sampling_rate=sr)
            return signal, sr

        with patch(
            "torch.hub.load", side_effect=_fake_torch_hub_load_factory(denoise)
        ):
            augmenter = AugmenterSilero(_make_df(paths))
            df_ret = augmenter.augment("all")

        new_files = list(df_ret.index.get_level_values(0))
        assert len(set(new_files)) == 3
        assert all(f not in paths for f in new_files)
