"""CQCC feature extraction helpers."""

import os

import numpy as np
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.feats_audio import read_indexed_audio, series_to_float_df
from nkululeko.feat_extract.featureset import Featureset

try:
    import librosa
    from scipy.fft import dct as _scipy_dct

    _LIBROSA_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _LIBROSA_AVAILABLE = False


N_CQCC = 40  # Number of CQCC coefficients.
N_CQT_BINS = 84  # CQT bins (7 octaves * 12 bins).
SR = 16000  # Sampling rate.
FRAME_PERIOD = 80  # Frame period.


class CqccFeatureExtractor:
    """Extract summary statistics from constant-Q cepstral coefficients."""

    def __init__(
        self,
        sample_rate,
        frame_period,
        n_cqcc=N_CQCC,
        n_cqt_bins=N_CQT_BINS,
    ):
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.n_cqcc = n_cqcc
        self.n_cqt_bins = n_cqt_bins
        self.available = _LIBROSA_AVAILABLE
        self._warned = False
        self.warning = (
            "WARNING: librosa not installed, skipping cqcc features. "
            "Install with: pip install librosa"
        )

    def extract(self, signal_tensor):
        """Return CQCC mean/std features for a mono signal tensor."""
        if not self.available:
            if not self._warned:
                print(self.warning)
                self._warned = True
            return {}

        if hasattr(signal_tensor, "cpu"):
            signal_np = signal_tensor.cpu().numpy().astype(np.float32)
        else:
            signal_np = np.asarray(signal_tensor, dtype=np.float32)
        cqt = librosa.cqt(
            signal_np,
            sr=self.sample_rate,
            n_bins=self.n_cqt_bins,
            hop_length=self.frame_period,
        )
        log_cqt = np.log(np.abs(cqt) + 1e-8)
        cqcc_matrix = _scipy_dct(log_cqt, axis=0, type=2, norm="ortho")[
            : self.n_cqcc
        ]

        emb = {}
        for i in range(cqcc_matrix.shape[0]):
            emb[f"cqcc_{i}_mean"] = np.mean(cqcc_matrix[i])
            emb[f"cqcc_{i}_std"] = np.std(cqcc_matrix[i])
        return emb


class CqccSet(Featureset):
    """Top-level feature set for `[FEATS] type = ['cqcc']`."""

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        self.frame_period = int(
            self.util.config_val("FEATS", "cqcc.frame_period", FRAME_PERIOD)
        )
        self.sample_rate = int(self.util.config_val("FEATS", "cqcc.sample_rate", SR))
        self.n_cqcc = int(self.util.config_val("FEATS", "cqcc.n_cqcc", N_CQCC))
        self.n_cqt_bins = int(
            self.util.config_val("FEATS", "cqcc.n_cqt_bins", N_CQT_BINS)
        )
        self.extractor = CqccFeatureExtractor(
            sample_rate=self.sample_rate,
            frame_period=self.frame_period,
            n_cqcc=self.n_cqcc,
            n_cqt_bins=self.n_cqt_bins,
        )

    def extract(self):
        """Extract CQCC features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting CQCC, this might take a while...")
            self.df = self._extract_index(self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted CQCC values")
            self.df = self.util.get_store(storage, store_format)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def extract_sample(self, signal, sr):
        if not self.extractor.available:
            if not self.extractor._warned:
                print(self.extractor.warning)
                self.extractor._warned = True
            return pd.DataFrame([{}]).to_numpy()
        feats = self.extractor.extract(signal)
        return pd.DataFrame([feats]).astype(float).to_numpy()

    def _extract_index(self, file_index):
        if not self.extractor.available:
            if not self.extractor._warned:
                print(self.extractor.warning)
                self.extractor._warned = True
            return pd.DataFrame(index=file_index)
        emb_series = pd.Series(index=file_index, dtype=object)
        skipped = 0
        for row_index in file_index.to_list():
            try:
                signal, _ = read_indexed_audio(row_index, self.sample_rate)
                emb_series[row_index] = self.extractor.extract(signal[0])
            except Exception as e:
                print(f"WARNING: featureset: skipping {row_index}: {e}")
                skipped += 1
        if skipped:
            print(
                f"WARNING: featureset: skipped {skipped} files that failed to load or extract CQCC features"
            )
        return series_to_float_df(emb_series)
