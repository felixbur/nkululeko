"""Unit tests for Resampler.resample() (nkululeko/augmenting/resampler.py).

GH #479: load_audio errors other than RuntimeError (audiofile raises e.g.
FileNotFoundError/ValueError, not torchaudio's RuntimeError) used to abort
the whole run instead of being caught; and a caught error left `signal`
undefined (first file) or stale from the previous file (later files), which
was then silently resampled/written under the failing file's name.

PR #485 review: with `replace=False`, the fix's `continue` skipped adding
anything to `new_files`, so the later index rebuild got fewer names than
rows and crashed; fixed by appending every row's resulting name (including
skipped ones) and rebuilding via `audformat.segmented_index()` instead of
the already-row/level-mismatched `set_levels()`.
"""

import audformat
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
    missing_path = str(tmp_path / "missing.wav")
    sf.write(good_path, np.zeros(100, dtype="float32"), SR)
    # the failing file is first: pre-fix, `signal` was never assigned yet,
    # so the next line (`signal.shape[0]`) raised UnboundLocalError
    df = pd.DataFrame(index=[missing_path, good_path])

    monkeypatch.setattr(
        "nkululeko.augmenting.resampler.load_audio",
        _fake_load_audio_factory(missing_path),
    )

    resampler = Resampler(df, replace=True, not_testing=True)
    resampler.resample()  # must not raise


def test_error_does_not_write_a_stale_signal_under_its_own_name(tmp_path, monkeypatch):
    """The second good file's content must be its own -- not a's, carried
    over from before the error because `signal` was never reassigned."""
    good_a = str(tmp_path / "a.wav")
    missing = str(tmp_path / "missing.wav")
    good_b = str(tmp_path / "b.wav")
    # a different rate than SAMPLING_RATE so each is actually resampled and
    # rewritten in place (replace=True), not just read and left alone
    sf.write(good_a, np.zeros(80, dtype="float32"), 8000)
    sf.write(good_b, np.full(80, 0.5, dtype="float32"), 8000)
    df = pd.DataFrame(index=[good_a, missing, good_b])

    debug_messages = []
    monkeypatch.setattr(
        "nkululeko.augmenting.resampler.load_audio",
        _fake_load_audio_factory(missing),
    )

    resampler = Resampler(df, replace=True, not_testing=True)
    resampler.util.debug = lambda msg: debug_messages.append(msg)
    resampler.resample()

    assert any("1 errors" in msg for msg in debug_messages)
    # "missing.wav" was never created (no stale write under its name)
    import os

    assert not os.path.exists(missing)
    b_signal, b_sr = sf.read(good_b)
    assert b_sr == SR
    # resampling filter ringing affects a few edge samples; the bulk of the
    # signal must still be b's own content (~0.5), not a's stale zeros
    assert np.mean(b_signal) == pytest.approx(0.5, abs=0.05)


def test_replace_false_rebuilds_the_index_after_a_skipped_file(tmp_path, monkeypatch):
    """GH #485 review: a skipped row must still get an entry in the rebuilt
    index, else segmented_index()/set_levels() gets fewer names than rows."""
    good_a = str(tmp_path / "a.wav")
    missing = str(tmp_path / "missing.wav")
    good_b = str(tmp_path / "b.wav")
    sf.write(good_a, np.zeros(80, dtype="float32"), 8000)
    sf.write(good_b, np.full(80, 0.5, dtype="float32"), 8000)

    df = pd.DataFrame(index=audformat.segmented_index([good_a, missing, good_b]))
    df["target"] = "anger"

    monkeypatch.chdir(tmp_path)  # resample() writes "resampled.csv" to cwd
    monkeypatch.setattr(
        "nkululeko.augmenting.resampler.load_audio",
        _fake_load_audio_factory(missing),
    )

    resampler = Resampler(df, replace=False, not_testing=True)
    resampler.resample()  # must not raise (index/new_files length mismatch)

    files = resampler.df.index.get_level_values("file").tolist()
    assert missing in files  # the skipped row keeps its original name
    assert any(f.endswith("a_16kHz.wav") for f in files)
    assert any(f.endswith("b_16kHz.wav") for f in files)
