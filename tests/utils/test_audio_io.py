"""load_audio (nkululeko/utils/audio_io.py): audiofile in place of
torchaudio.load(), which needs torchcodec in torchaudio >= 2.9."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from nkululeko.utils.audio_io import load_audio

SR = 16000


@pytest.fixture
def wav(tmp_path):
    """1 s of a ramp, mono, 16 kHz."""
    path = tmp_path / "a b.wav"  # a space in the name
    signal = (np.arange(SR) / SR - 0.5).astype("float32")
    sf.write(path, signal, SR, subtype="FLOAT")
    return str(path), signal


def test_whole_file_is_a_channels_by_samples_float32_tensor(wav):
    path, signal = wav
    out, sr = load_audio(path)
    assert sr == SR
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    assert out.shape == (1, SR)
    np.testing.assert_allclose(out[0].numpy(), signal)


def test_segment_between_start_and_end(wav):
    path, signal = wav
    out, sr = load_audio(path, pd.Timedelta(seconds=0.25), pd.Timedelta(seconds=0.75))
    assert sr == SR and out.shape == (1, SR // 2)
    np.testing.assert_allclose(out[0].numpy(), signal[SR // 4 : 3 * SR // 4])


def test_a_segment_beyond_the_end_stops_at_the_end(wav):
    path, signal = wav
    out, _ = load_audio(path, pd.Timedelta(seconds=0.5), pd.Timedelta(seconds=3))
    assert out.shape == (1, SR // 2)


def test_start_only_reads_to_the_end(wav):
    path, signal = wav
    out, _ = load_audio(path, pd.Timedelta(seconds=0.5))
    np.testing.assert_allclose(out[0].numpy(), signal[SR // 2 :])


def test_stereo_keeps_its_channels(tmp_path):
    path = tmp_path / "s.wav"
    data = np.stack([np.full(100, 0.25), np.full(100, -0.25)], axis=1)
    sf.write(path, data, SR, subtype="FLOAT")
    out, _ = load_audio(str(path))
    assert out.shape == (2, 100)
    np.testing.assert_allclose(out[:, 0].numpy(), [0.25, -0.25])


def test_16_bit_files_are_normalised_to_one(tmp_path):
    path = tmp_path / "p.wav"
    sf.write(path, np.array([0.5, -0.5], dtype="float32"), SR, subtype="PCM_16")
    out, _ = load_audio(str(path))
    np.testing.assert_allclose(out[0].numpy(), [0.5, -0.5], atol=1e-4)


def test_the_result_can_be_changed_in_place(wav):
    out, _ = load_audio(wav[0])
    out *= 2  # no read-only numpy buffer behind it


def test_nat_end_reads_the_whole_file(wav):
    """GH #481: a segmented index's whole-file rows carry end=pd.NaT, not
    None; `pd.NaT is not None`, so it used to reach
    (end - start).total_seconds() and read with duration=nan."""
    path, signal = wav
    out, sr = load_audio(path, pd.NaT, pd.NaT)
    assert sr == SR
    np.testing.assert_allclose(out[0].numpy(), signal)


def test_nat_start_with_real_end(wav):
    path, signal = wav
    out, _ = load_audio(path, pd.NaT, pd.Timedelta(seconds=0.5))
    np.testing.assert_allclose(out[0].numpy(), signal[: SR // 2])
