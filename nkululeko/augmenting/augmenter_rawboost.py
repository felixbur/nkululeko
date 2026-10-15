"""augmenter_rawboost.py

RawBoost (Tak et al., ICASSP 2022) as an augmentation method: writes a
noisy copy of every selected audio file. Because it only produces audio
files, the result can be used for training any model (see
`nkululeko/augmenting/rawboost.py` for the algorithm and its settings).
"""

import hashlib
import os
from dataclasses import asdict

import audeer
import audiofile
import numpy as np
import pandas as pd
from tqdm import tqdm

from nkululeko.augmenting.rawboost import RawBoostParams, apply_rawboost
from nkululeko.utils.dataframe import remap_augmented_index, should_reuse_file
from nkululeko.utils.files import mirror_relpath
from nkululeko.utils.util import Util

# audiofile can write these; anything else (e.g. mp3) is written as wav
_WRITABLE = (".wav", ".flac", ".ogg")


def rawboost_signal(signal, sr, params):
    """RawBoost for audiofile's (samples,) or (channels, samples) array: every
    channel gets its own noise. The result is scaled down if it exceeds the
    range of an audio file (the additive noise can push it over 1)."""
    signal = np.asarray(signal)
    if signal.ndim == 1:
        out = apply_rawboost(signal, sr, params)
    else:
        out = np.stack([apply_rawboost(channel, sr, params) for channel in signal])
    peak = np.amax(np.abs(out)) if out.size else 0.0
    if peak > 1:
        out = out / peak
    return out.astype(np.float32)


class AugmenterRawBoost:
    """Add RawBoost noise to the selected samples."""

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter_rawboost")
        self.params = RawBoostParams.from_util(self.util)

    def _out_dir(self):
        """Store folder named after the settings, so that files made with
        other settings are not reused (`DATA.no_reuse` is not needed)."""
        digest = hashlib.sha1(repr(sorted(asdict(self.params).items())).encode())
        name = f"algo{self.params.algo}-{digest.hexdigest()[:6]}"
        return os.path.join(self.util.get_path("store"), "rawboost", name)

    def augment(self, sample_selection):
        """Write the noisy files and return a dataframe with the new files
        as index."""
        # dedupe: a segmented index can list the same file for multiple
        # (start, end) rows, and the noise is added per file
        files = pd.unique(self.df.index.get_level_values(0).values)
        filepath = self._out_dir()
        audeer.mkdir(filepath)
        self.util.debug(
            f"adding RawBoost (algorithm {self.params.algo}) to "
            f"{sample_selection} samples, writing to {filepath}"
        )
        index_map = {}
        reused = 0
        for f in tqdm(files):
            # keyed by the full source path so that files of different
            # datasets with the same name do not collide
            new_full_name = os.path.join(filepath, mirror_relpath(f))
            base, ext = os.path.splitext(new_full_name)
            if ext.lower() not in _WRITABLE:
                new_full_name = base + ".wav"
            if should_reuse_file(self.util, "DATA", new_full_name):
                reused += 1
                index_map[f] = new_full_name
                continue
            audeer.mkdir(os.path.dirname(new_full_name))
            signal, sr = audiofile.read(f)
            audiofile.write(
                new_full_name,
                signal=rawboost_signal(signal, sr, self.params),
                sampling_rate=sr,
            )
            index_map[f] = new_full_name
        if reused:
            self.util.debug(f"reused {reused}/{len(files)} previously written files")
        return remap_augmented_index(self.df, index_map)
