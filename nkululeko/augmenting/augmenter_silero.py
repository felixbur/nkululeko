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

from nkululeko.utils.util import Util


class AugmenterSilero:
    """Denoise the selected samples with Silero's speech enhancement model."""

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter_silero")
        model_name = self.util.config_val("AUGMENT", "silero_model", "small_fast")
        device = self.util.config_val("MODEL", "device", "cpu")
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.util.debug(
            f"loading silero denoise model '{model_name}' on {self.device}"
        )
        self.model, _, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
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
        for f in tqdm(files):
            filename = os.path.basename(f)
            parent = os.path.dirname(f).split("/")[-1]
            newpath = f"{filepath}/{parent}/"
            audeer.mkdir(newpath)
            new_full_name = newpath + filename
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
        df_ret = self.df.copy()

        file_index = df_ret.index.to_series().map(lambda x: index_map[x[0]]).values
        # workaround because i just couldn't get this easier...
        arrays = [
            file_index,
            list(df_ret.index.get_level_values(1)),
            list(df_ret.index.get_level_values(2)),
        ]
        new_index = pd.MultiIndex.from_arrays(arrays, names=("file", "start", "end"))
        df_ret = df_ret.set_index(new_index)

        return df_ret
