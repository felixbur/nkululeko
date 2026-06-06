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
        self.warning = (
            "WARNING: librosa not installed, skipping cqcc features. "
            "Install with: pip install librosa"
        )

    def extract(self, signal_tensor):
        """Return CQCC mean/std features for a mono signal tensor."""
        if not self.available:
            print(self.warning)
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
