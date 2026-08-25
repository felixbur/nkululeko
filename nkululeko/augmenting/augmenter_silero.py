"""augmenter_silero.py

Speech enhancement / noise removal implemented with Silero's denoise model
-> https://github.com/snakers4/silero-models

Used as an augmentation method: denoises the selected samples and adds them
as additional (or alternative) training data, addressing
https://github.com/felixbur/nkululeko/issues/215
"""

import os

import audeer
import audiofile
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from nkululeko.utils.dataframe import remap_augmented_index, should_reuse_file
from nkululeko.utils.util import Util

# Pinned so results are reproducible run-to-run instead of tracking
# whatever torch.hub.load would otherwise resolve as the repo's default
# branch. Bump deliberately when a newer Silero release is desired.
SILERO_MODELS_REF = "v5.6"


class AugmenterSilero:
    """Denoise the selected samples with Silero's speech enhancement model."""

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter_silero")
        model_name = self.util.config_val("AUGMENT", "silero_model", "small_slow")
        device = self.util.config_val("MODEL", "device", "cpu")
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.util.debug(
            f"loading silero denoise model '{model_name}' on {self.device}"
        )
        self.model, _, utils = torch.hub.load(
            repo_or_dir=f"snakers4/silero-models:{SILERO_MODELS_REF}",
            model="silero_denoise",
            name=model_name,
            device=self.device,
        )
        _, _, self.denoise = utils

    def augment(self, sample_selection):
        """
        denoise the selected files and return a dataframe with new files index.
        """
        # dedupe: a segmented index can list the same file for multiple
        # (start, end) rows, and denoising is a per-file operation
        files = pd.unique(self.df.index.get_level_values(0).values)
        store = self.util.get_path("store")
        filepath = f"{store}silero/"
        audeer.mkdir(filepath)
        self.util.debug(f"denoising {sample_selection} samples to {filepath}")
        index_map = {}
        reused = 0
        for f in tqdm(files):
            # Keyed by the full source path (not just the immediate parent
            # directory name) so two datasets that happen to share a
            # subfolder name (e.g. both using "wav/") and a filename don't
            # collide and silently overwrite each other's denoised file.
            rel = os.path.abspath(f).lstrip(os.sep)
            new_full_name = os.path.join(filepath, rel)
            audeer.mkdir(os.path.dirname(new_full_name))
            if should_reuse_file(self.util, "DATA", new_full_name):
                # denoising (especially the small_slow model) is slow, so
                # skip files already denoised by a previous run
                reused += 1
                index_map[f] = new_full_name
                continue
            org_sr = audiofile.sampling_rate(f)
            _, out_sr = self.denoise(self.model, f, new_full_name, device=self.device)
            if out_sr != org_sr:
                # keep the denoised file at the original sampling rate so it
                # stays consistent with the rest of the (unaugmented) samples
                signal, _ = audiofile.read(new_full_name)
                signal_t = torch.as_tensor(signal).reshape(1, -1)
                resampler = torchaudio.transforms.Resample(out_sr, org_sr)
                signal = resampler(signal_t).squeeze(0).numpy()
                audiofile.write(new_full_name, signal=signal, sampling_rate=org_sr)
            index_map[f] = new_full_name
        if reused:
            self.util.debug(
                f"reused {reused}/{len(files)} previously denoised files"
            )

        return remap_augmented_index(self.df, index_map)
