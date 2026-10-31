"""Audio loading for the feature extractors.

`torchaudio.load()` needs the optional `torchcodec` package (and a system
FFmpeg) since torchaudio 2.9, which nkululeko does not depend on. The audio
is read with `audiofile` instead, which every other part of nkululeko
already uses, and handed over as the (channels, samples) float32 tensor that
`torchaudio.load()` returned.
"""

import audiofile
import pandas as pd
import torch


def load_audio(file, start=None, end=None):
    """Read `file` (or only its segment from `start` to `end`) and return
    (signal, sampling_rate): `signal` is a float32 tensor of shape
    (channels, samples) in [-1, 1], like the one of `torchaudio.load()`.

    start, end: pandas Timedelta (or anything with `total_seconds()`) or
        None/NaT; a segment reaching beyond the end of the file stops there.
        With `start` only, the audio is read from there to the end of the
        file. A segmented index's whole-file rows carry `end = pd.NaT`
        (`end is not None` but `pd.isna(end)`), so that must be checked
        too, else `(end - start).total_seconds()` is NaN.
    """
    start = None if pd.isna(start) else start
    end = None if pd.isna(end) else end
    offset = None if start is None else start.total_seconds()
    duration = None
    if end is not None:
        duration = (
            end - (start if start is not None else pd.Timedelta(0))
        ).total_seconds()
    signal, sampling_rate = audiofile.read(
        file, offset=offset, duration=duration, always_2d=True
    )
    return torch.from_numpy(signal.copy()), sampling_rate
