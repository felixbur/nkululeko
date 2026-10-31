"""Unit tests for Resampler.resample() (nkululeko/augmenting/resampler.py).

GH #479: load_audio errors other than RuntimeError (audiofile raises e.g.
FileNotFoundError/ValueError, not torchaudio's RuntimeError) used to abort
the whole run instead of being caught; and a caught error left `signal`
undefined (first file) or stale from the previous file (later files),
which was then silently resampled/written under the failing file's name.
"""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from nkululeko.augmenting.resampler import Resampler

SR = 16000


def _fake_load_audio_factory(bad_file):
    def fake_load_audio(f):
        if f == bad_file:
            raise FileNotFoundError(f"no such file: {f}")
        signal, sr = sf.read(f, dtype="float32")
        return torch.from_numpy(signal).unsqueeze(0), sr

    return fake_load_audio


def test_error_on_first_file_does_not_leave_signal_undefined(tmp_path, monkeypatch):
    good_path = str(tmp_path / "good.wav")
    sf.write(good_path, np.zeros(100, dtype="float32"), SR)
    # the failing file is first: pre-fix, `signal` was never assigned yet,
    # so the next line (`signal.shape[0]`) raised UnboundLocalError
    df = pd.DataFrame(index=["missing.wav", good_path])

    monkeypatch.setattr(
        "nkululeko.augmenting.resampler.load_audio",
        _fake_load_audio_factory("missing.wav"),
    )

    resampler = Resampler(df, replace=True, not_testing=True)
    resampler.resample()  # must not raise


def test_error_counted_and_processing_continues_after_it(tmp_path, monkeypatch):
    good_a = str(tmp_path / "a.wav")
    good_b = str(tmp_path / "b.wav")
    sf.write(good_a, np.zeros(100, dtype="float32"), SR)
    sf.write(good_b, np.ones(100, dtype="float32") * 0.5, SR)
    # the failure sits between two good files: pre-fix, the second good
    # file would silently reuse the first good file's stale `signal`
    df = pd.DataFrame(index=[good_a, "missing.wav", good_b])

    debug_messages = []
    monkeypatch.setattr(
        "nkululeko.augmenting.resampler.load_audio",
        _fake_load_audio_factory("missing.wav"),
    )

    resampler = Resampler(df, replace=True, not_testing=True)
    resampler.util.debug = lambda msg: debug_messages.append(msg)
    resampler.resample()

    assert any("1 errors" in msg for msg in debug_messages)
