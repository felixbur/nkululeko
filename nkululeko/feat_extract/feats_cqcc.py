"""CQCC feature extraction helpers."""

import numpy as np

try:
    import librosa
    from scipy.fft import dct as _scipy_dct

    _LIBROSA_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    _LIBROSA_AVAILABLE = False


N_CQCC = 40  # Number of CQCC coefficients.
N_CQT_BINS = 84  # CQT bins (7 octaves * 12 bins).


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
            if not getattr(self, "_warned", False):
                print(self.warning)
                self._warned = True
            return {}

        signal_np = signal_tensor.cpu().numpy().astype(np.float32)
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


class CqccSet:
    """Top-level feature set wrapper for `[FEATS] type = ['cqcc']`."""

    def __init__(self, name, data_df, feats_type):
        from nkululeko.feat_extract.feats_sptk import SptkSet

        self._sptk = SptkSet(name, data_df, feats_type)
        self._sptk.features_requested = ["cqcc"]

    @property
    def df(self):
        return self._sptk.df

    @df.setter
    def df(self, value):
        self._sptk.df = value

    def extract(self):
        return self._sptk.extract()

    def filter(self):
        return self._sptk.filter()

    def extract_sample(self, signal, sr):
        return self._sptk.extract_sample(signal, sr)
