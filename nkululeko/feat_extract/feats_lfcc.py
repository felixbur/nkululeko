"""LFCC feature extraction helpers."""

import os

import numpy as np
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import SAMPLING_RATE
from nkululeko.feat_extract.feats_audio import read_indexed_audio, series_to_float_df
from nkululeko.feat_extract.featureset import Featureset

try:
    import torch
    import torchaudio.transforms as T_audio

    _TORCHAUDIO_LFCC = True
except (ImportError, AttributeError, OSError):
    _TORCHAUDIO_LFCC = False


# All values below are overridable via the [FEATS] section of the INI file
# (e.g. lfcc.n_lfcc, lfcc.frame_length); they only serve as defaults.
N_LFCC = 40  # Number of LFCC coefficients.
FRAME_LENGTH = 400  # Frame length.
FRAME_PERIOD = 80  # Frame period.
N_FFT = 512  # FFT length.


class LfccFeatureExtractor:
    """Extract summary statistics from linear frequency cepstral coefficients."""

    # Class-level default so the flag is readable even on instances built
    # without __init__ (e.g. unit tests using __new__).
    _warned = False

    def __init__(
        self,
        sample_rate,
        frame_length,
        frame_period,
        fft_length,
        device="cpu",
        n_lfcc=N_LFCC,
    ):
        self.available = False
        self.warning = (
            "WARNING: torchaudio LFCC not available (requires torchaudio>=0.11), "
            "skipping lfcc features"
        )

        if not _TORCHAUDIO_LFCC:
            return

        try:
            self.transform = T_audio.LFCC(
                sample_rate=sample_rate,
                n_lfcc=n_lfcc,
                speckwargs={
                    "n_fft": fft_length,
                    "win_length": frame_length,
                    "hop_length": frame_period,
                },
            ).to(device)
            self.available = True
        except Exception:
            self.available = False

    def warn_unavailable(self):
        """Print the dependency warning once when the extractor is unavailable."""
        if not self._warned:
            print(self.warning)
            self._warned = True

    def extract(self, signal_tensor):
        """Return LFCC mean/std features for a mono signal tensor."""
        if not self.available:
            self.warn_unavailable()
            return {}

        # LFCC expects (channel, time); output is (channel, n_lfcc, frames).
        lfcc_out = self.transform(signal_tensor.unsqueeze(0))
        lfcc_np = lfcc_out.squeeze(0).cpu().numpy()

        emb = {}
        for i in range(lfcc_np.shape[0]):
            emb[f"lfcc_{i}_mean"] = np.mean(lfcc_np[i])
            emb[f"lfcc_{i}_std"] = np.std(lfcc_np[i])
        return emb


class LfccSet(Featureset):
    """Top-level feature set for `[FEATS] type = ['lfcc']`."""

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        cuda = "cuda" if _TORCHAUDIO_LFCC and torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.frame_length = int(
            self.util.config_val("FEATS", "lfcc.frame_length", FRAME_LENGTH)
        )
        self.frame_period = int(
            self.util.config_val("FEATS", "lfcc.frame_period", FRAME_PERIOD)
        )
        self.fft_length = int(self.util.config_val("FEATS", "lfcc.fft_length", N_FFT))
        self.sample_rate = int(
            self.util.config_val("FEATS", "lfcc.sample_rate", SAMPLING_RATE)
        )
        self.n_lfcc = int(self.util.config_val("FEATS", "lfcc.n_lfcc", N_LFCC))
        self.extractor = LfccFeatureExtractor(
            sample_rate=self.sample_rate,
            frame_length=self.frame_length,
            frame_period=self.frame_period,
            fft_length=self.fft_length,
            device=self.device,
            n_lfcc=self.n_lfcc,
        )

    def extract(self):
        """Extract LFCC features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting LFCC, this might take a while...")
            self.df = self._extract_index(self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted LFCC values")
            self.df = self.util.get_store(storage, store_format)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def extract_sample(self, signal, sr):
        if not self.extractor.available:
            self.extractor.warn_unavailable()
            return pd.DataFrame([{}]).to_numpy()
        signal_tensor = torch.tensor(signal, device=self.device).float()
        feats = self.extractor.extract(signal_tensor)
        return pd.DataFrame([feats]).astype(float).to_numpy()

    def _extract_index(self, file_index):
        if not self.extractor.available:
            self.extractor.warn_unavailable()
            return pd.DataFrame(index=file_index)
        emb_series = pd.Series(index=file_index, dtype=object)
        skipped = 0
        for row_index in file_index.to_list():
            try:
                signal, _ = read_indexed_audio(row_index, self.sample_rate)
                signal_tensor = torch.tensor(signal[0], device=self.device).float()
                emb_series[row_index] = self.extractor.extract(signal_tensor)
            except Exception as e:
                print(f"WARNING: featureset: skipping {row_index}: {e}")
                skipped += 1
        if skipped:
            print(
                f"WARNING: featureset: skipped {skipped} files that failed to load or extract LFCC features"
            )
        return series_to_float_df(emb_series)
