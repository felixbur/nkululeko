"""
feats_spectra.py

Inspired by code from Su Lei

"""

import os
import pathlib

import audeer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import SAMPLING_RATE
from nkululeko.feat_extract.featureset import Featureset


class Spectraloader(Featureset):
    def __init__(self, name, data_df, feat_type):
        """Constructor setting the name"""
        super().__init__(name, data_df, feat_type)
        self.sampling_rate = SAMPLING_RATE
        self.num_bands = int(self.util.config_val("FEATS", "fft_nbands", "64"))
        self.win_dur = int(self.util.config_val("FEATS", "fft_win_dur", "25"))
        self.hop_dur = int(self.util.config_val("FEATS", "fft_hop_dur", "10"))

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting mel spectra, this might take a while...")
            image_store = audeer.mkdir(f"{store}{self.name}")
            images = []
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = torchaudio.load(
                    file,
                    frame_offset=int(start.total_seconds() * 16000),
                    num_frames=int((end - start).total_seconds() * 16000),
                )
                assert sampling_rate == 16000, f"got {sampling_rate} instead of 16000"
                image = self._waveform2rgb(signal)
                outfile = f"{image_store}/{pathlib.Path(file).stem}_{idx}.jpg"
                image.save(outfile)
                images.append(outfile)
            self.df = pd.DataFrame(images, index=self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted spectrograms")
            self.df = self.util.get_store(storage, store_format)

    def _waveform2rgb(self, waveform, target_size=(256, 256)):
        # Transform to spectrogram
        spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_mels=self.num_bands,
            hop_length=int(self.hop_dur * SAMPLING_RATE / 1000),
            win_length=int(self.win_dur * SAMPLING_RATE / 1000),
        )(waveform)
        melspec = T.AmplitudeToDB()(spectrogram)[0].numpy()
        melspec_norm = (melspec - np.min(melspec)) / (np.max(melspec) - np.min(melspec))

        # Map normalized Mel spectrogram to viridis colormap
        cmapped = plt.get_cmap("viridis")(melspec_norm)

        # Convert this colormap representation to a format suitable for creating a PIL Image
        image_array = (cmapped[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode="RGB")
        image = ImageOps.flip(image)

        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image
