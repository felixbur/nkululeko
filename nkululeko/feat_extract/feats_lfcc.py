"""LFCC feature extraction helpers."""

import numpy as np

try:
    import torchaudio.transforms as T_audio

    _TORCHAUDIO_LFCC = True
except (ImportError, AttributeError, OSError):
    _TORCHAUDIO_LFCC = False


N_LFCC = 40  # Number of LFCC coefficients.


class LfccFeatureExtractor:
    """Extract summary statistics from linear frequency cepstral coefficients."""

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
        self._warned = False
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

    def extract(self, signal_tensor):
        """Return LFCC mean/std features for a mono signal tensor."""
        if not self.available:
            if not getattr(self, "_warned", False):
                print(self.warning)
                self._warned = True
            return {}

        # LFCC expects (channel, time); output is (channel, n_lfcc, frames).
        lfcc_out = self.transform(signal_tensor.unsqueeze(0))
        lfcc_np = lfcc_out.squeeze(0).cpu().numpy()

        emb = {}
        for i in range(lfcc_np.shape[0]):
            emb[f"lfcc_{i}_mean"] = np.mean(lfcc_np[i])
            emb[f"lfcc_{i}_std"] = np.std(lfcc_np[i])
        return emb
