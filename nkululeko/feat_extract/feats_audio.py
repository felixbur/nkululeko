"""Shared audio helpers for feature extractors."""

import subprocess

import audiofile
import numpy as np
import pandas as pd


def read_indexed_audio(row_index, sample_rate):
    """Read a file path or segmented audformat-style index tuple."""
    if isinstance(row_index, tuple) and len(row_index) == 3:
        file, start, end = row_index
        return read_audio(
            file,
            sample_rate=sample_rate,
            offset=start.total_seconds(),
            duration=(end - start).total_seconds(),
        )
    file = row_index if isinstance(row_index, str) else str(row_index)
    return read_audio(file, sample_rate=sample_rate)


def read_audio(file, sample_rate, offset=None, duration=None):
    """Read audio with an ffmpeg fallback for files unsupported by libsndfile."""
    try:
        kwargs = {"always_2d": True}
        if offset is not None:
            kwargs["offset"] = offset
        if duration is not None:
            kwargs["duration"] = duration
        signal, sampling_rate = audiofile.read(file, **kwargs)
        if signal.shape[0] > 1:
            signal = signal.mean(axis=0, keepdims=True)
        return signal, sampling_rate
    except Exception:
        cmd = ["ffmpeg", "-v", "quiet", "-i", file]
        if offset is not None:
            cmd += ["-ss", str(offset)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += [
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or len(result.stdout) == 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {file}: {result.stderr.decode()}"
            )
        audio = (
            np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return audio[np.newaxis, :], sample_rate


def series_to_float_df(emb_series):
    """Convert a series of feature dictionaries to a float DataFrame."""
    valid = emb_series.notna()
    if not valid.all():
        emb_series = emb_series[valid]
    df = pd.DataFrame(emb_series.values.tolist(), index=emb_series.index)
    for col in df.columns:
        if df[col].isnull().values.any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val if not np.isnan(mean_val) else 0)
    return df.astype(float)
