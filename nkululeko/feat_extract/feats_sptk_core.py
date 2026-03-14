"""Core SPTK feature extraction functions.

This module contains the core logic for extracting SPTK (Speech Signal Processing Toolkit)
features from audio signals using the diffsptk library.

https://github.com/sp-nitech/diffsptk
"""

import subprocess

import numpy as np
import pandas as pd
import torch
import audiofile
from tqdm import tqdm

import diffsptk

# Constants
FRAME_LENGTH = 400  # Frame length.
FRAME_PERIOD = 80  # Frame period.
N_FFT = 512  # FFT length.
M_DIM = 24  # Mel-cepstrum dimensions.
SR = 16000  # Sampling rate.
N_CHANNEL = 128


class SptkFeatureExtractor:
    """Class to extract SPTK features from audio signals."""

    def __init__(
        self,
        frame_length=FRAME_LENGTH,
        frame_period=FRAME_PERIOD,
        fft_length=N_FFT,
        n_channel=N_CHANNEL,
        sample_rate=SR,
        device="cpu",
    ):
        """Initialize SPTK feature extractors.

        Args:
            frame_length: Frame length for STFT
            frame_period: Frame period for STFT
            fft_length: FFT length
            n_channel: Number of channels for filterbank
            sample_rate: Sampling rate
            device: Device to use for computation ('cpu' or 'cuda')
        """
        self.frame_length = frame_length
        self.frame_period = frame_period
        self.fft_length = fft_length
        self.n_channel = n_channel
        self.sample_rate = sample_rate
        self.device = device

        # Initialize STFT
        self.stft = diffsptk.STFT(
            frame_length=self.frame_length,
            frame_period=self.frame_period,
            fft_length=self.fft_length,
            out_format="power",
        ).to(self.device)

        # Initialize filterbank
        self.fbank = diffsptk.FBANK(
            fft_length=self.fft_length,
            n_channel=self.n_channel,
            sample_rate=self.sample_rate,
        ).to(self.device)

        # Initialize mel-cepstral analysis
        self.mcep = diffsptk.MelCepstralAnalysis(
            fft_length=self.fft_length,
            cep_order=M_DIM,
            alpha=diffsptk.get_alpha(self.sample_rate),
            n_iter=10,
        ).to(self.device)

        # Initialize chroma filterbank
        self.chroma = diffsptk.ChromaFilterBankAnalysis(
            fft_length=self.fft_length,
            n_channel=self.n_channel,
            sample_rate=self.sample_rate,
            device=self.device,
        )

        # Initialize pitch extractor with error handling
        self.pitch_fallback = False
        self.pitch_features_available = False

        try:
            # Try default pitch algorithm first
            self.pitch = diffsptk.Pitch(
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                f_min=80,
                f_max=180,
                voicing_threshold=0.4,
                out_format="f0",
            )
            self.pitch_fallback = False
        except (ImportError, ModuleNotFoundError, AttributeError, Exception):
            # Fall back to PyWorld-based pitch extraction
            try:
                import pyworld as pw

                self.pitch_fallback = True
                self.pw = pw
            except ImportError:
                self.pitch_fallback = True
                self.pitch_features_available = False
                return

        # Initialize pitch-dependent features
        try:
            self.ap = diffsptk.Aperiodicity(
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                fft_length=self.fft_length,
                algorithm="d4c",
                out_format="a",
            )
            self.pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                fft_length=1024,
                algorithm="cheap-trick",
                out_format="power",
            )
            self.pitch_features_available = True
        except (ImportError, ModuleNotFoundError, AttributeError):
            self.pitch_features_available = False
            self.pitch_fallback = True

    def extract_features_from_signal(self, signal_tensor, features_requested=None):
        """Extract SPTK features from a signal tensor.

        Args:
            signal_tensor: Audio signal as torch tensor
            features_requested: List of features to extract (default: ['stft', 'fbank'])

        Returns:
            Dictionary of extracted features
        """
        if features_requested is None:
            features_requested = ["stft", "fbank"]

        # Pad signal if too short
        if signal_tensor.shape[0] < self.frame_period:
            pad_width = self.frame_period - signal_tensor.shape[0]
            signal_tensor = torch.nn.functional.pad(
                signal_tensor, (0, pad_width), mode="constant", value=0
            )

        # Extract STFT (needed for other features)
        stft_features = self.stft(signal_tensor)

        emb = {}

        # Extract requested features
        if "stft" in features_requested:
            stft_np = stft_features.cpu().numpy()
            # Per-bin statistics for rich phase/spectral information
            # stft_np shape: (n_frames, fft_length/2 + 1)
            if stft_np.ndim >= 2:
                n_bins = stft_np.shape[-1]
                for i in range(n_bins):
                    bin_data = stft_np[..., i]
                    emb[f"stft_{i}_mean"] = np.mean(bin_data)
                    emb[f"stft_{i}_std"] = np.std(bin_data)
            else:
                # Fallback for unexpected shapes
                emb["stft_mean"] = np.mean(stft_np)
                emb["stft_std"] = np.std(stft_np)
                emb["stft_min"] = np.min(stft_np)
                emb["stft_max"] = np.max(stft_np)

        if "fbank" in features_requested:
            fbank_features = self.fbank(stft_features)
            fbank_np = fbank_features.cpu().numpy()
            for i in range(fbank_np.shape[-1]):
                channel_data = fbank_np[..., i]
                emb[f"fbank_{i}_mean"] = np.mean(channel_data)
                emb[f"fbank_{i}_std"] = np.std(channel_data)

        if "mcep" in features_requested:
            mcep_features = self.mcep(stft_features)
            mcep_np = mcep_features.cpu().numpy()
            for i in range(mcep_np.shape[-1]):
                coef_data = mcep_np[..., i]
                emb[f"mcep_{i}_mean"] = np.mean(coef_data)
                emb[f"mcep_{i}_std"] = np.std(coef_data)

        if "chroma" in features_requested:
            chroma_features = self.chroma(stft_features)
            chroma_np = chroma_features.cpu().numpy()
            for i in range(chroma_np.shape[-1]):
                channel_data = chroma_np[..., i]
                emb[f"chroma_{i}_mean"] = np.mean(channel_data)
                emb[f"chroma_{i}_std"] = np.std(channel_data)

        # Extract pitch-dependent features if available
        if self.pitch_features_available and not self.pitch_fallback:
            try:
                if self.pitch_fallback:
                    # Use PyWorld for pitch extraction
                    signal_np = signal_tensor.cpu().numpy().astype(np.float64)
                    f0, _ = self.pw.dio(
                        signal_np,
                        self.sample_rate,
                        frame_period=self.frame_period / self.sample_rate * 1000,
                    )
                    f0 = self.pw.stonemask(signal_np, f0, _, self.sample_rate)
                    pitch_features = torch.tensor(f0, device=self.device)
                else:
                    pitch_features = self.pitch(signal_tensor)

                ap_features = self.ap(signal_tensor)
                pitch_spec_features = self.pitch_spec(signal_tensor)

                pitch_np = (
                    pitch_features.cpu().numpy()
                    if isinstance(pitch_features, torch.Tensor)
                    else pitch_features
                )
                ap_np = ap_features.cpu().numpy()
                pitch_spec_np = pitch_spec_features.cpu().numpy()

                emb.update(
                    {
                        "pitch_mean": np.mean(pitch_np),
                        "pitch_std": np.std(pitch_np),
                        "pitch_min": np.min(pitch_np),
                        "pitch_max": np.max(pitch_np),
                        "ap_mean": np.mean(ap_np),
                        "ap_std": np.std(ap_np),
                        "pitch_spec_mean": np.mean(pitch_spec_np),
                        "pitch_spec_std": np.std(pitch_spec_np),
                    }
                )
            except Exception:
                pass  # Skip pitch features if extraction fails

        return emb


def _read_audio(file, offset=None, duration=None):
    """Read audio file with fallback to ffmpeg for formats unsupported by libsndfile.

    Some FLAC files (e.g. ASVspoof 2021 with GSM/PSTN codec simulation) cannot be
    decoded by libsndfile and raise "unknown error in flac decoder". ffmpeg handles
    these files correctly.

    Returns:
        Tuple of (signal_2d, sample_rate) where signal_2d has shape (1, n_samples).
    """
    try:
        kwargs = {"always_2d": True}
        if offset is not None:
            kwargs["offset"] = offset
        if duration is not None:
            kwargs["duration"] = duration
        signal, sampling_rate = audiofile.read(file, **kwargs)
        # Downmix to mono if multi-channel
        if signal.shape[0] > 1:
            signal = signal.mean(axis=0, keepdims=True)
        return signal, sampling_rate
    except Exception:
        # Fall back to ffmpeg for files libsndfile cannot decode
        cmd = ["ffmpeg", "-v", "quiet", "-i", file]
        if offset is not None:
            cmd += ["-ss", str(offset)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", "-"]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or len(result.stdout) == 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {file}: {result.stderr.decode()}"
            )
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio[np.newaxis, :], SR  # Return shape (1, n_samples)


def compute_features(
    file_index,
    frame_length=FRAME_LENGTH,
    frame_period=FRAME_PERIOD,
    fft_length=N_FFT,
    n_channel=N_CHANNEL,
    sample_rate=SR,
    features_requested=None,
    device="cpu",
):
    """Compute SPTK features for multiple audio files.

    Args:
        file_index: Index of audio files (can be file paths or tuples of (file, start, end))
        frame_length: Frame length for STFT
        frame_period: Frame period for STFT
        fft_length: FFT length
        n_channel: Number of channels for filterbank
        sample_rate: Sampling rate
        features_requested: List of features to extract
        device: Device to use for computation

    Returns:
        DataFrame with extracted features
    """
    if features_requested is None:
        features_requested = ["stft", "fbank"]

    # Initialize extractor
    extractor = SptkFeatureExtractor(
        frame_length=frame_length,
        frame_period=frame_period,
        fft_length=fft_length,
        n_channel=n_channel,
        sample_rate=sample_rate,
        device=device,
    )

    emb_series = pd.Series(index=file_index, dtype=object)
    length = len(file_index)

    # Extract features for each file
    skipped = 0
    for idx, row_index in enumerate(tqdm(file_index.to_list(), total=length)):
        # Handle both tuple (file, start, end) and string file formats
        try:
            if isinstance(row_index, tuple) and len(row_index) == 3:
                file, start, end = row_index
                signal, sampling_rate = _read_audio(
                    file,
                    offset=start.total_seconds(),
                    duration=(end - start).total_seconds(),
                )
            else:
                # Single file path - read entire file
                file = row_index if isinstance(row_index, str) else str(row_index)
                signal, sampling_rate = _read_audio(file)

            # Convert to tensor
            signal_tensor = torch.tensor(signal[0], device=device).float()

            # Extract features
            emb = extractor.extract_features_from_signal(
                signal_tensor, features_requested
            )
            emb_series[row_index] = emb
        except Exception as e:
            print(f"WARNING: featureset: skipping {file}: {e}")
            skipped += 1

    if skipped:
        print(
            f"WARNING: featureset: skipped {skipped} files that failed to load or extract features"
        )

    # Convert to DataFrame, dropping any files that failed
    valid = emb_series.notna()
    if not valid.all():
        emb_series = emb_series[valid]
    df = pd.DataFrame(emb_series.values.tolist(), index=emb_series.index)

    # Fill NaN values with mean
    for col in df.columns:
        if df[col].isnull().values.any():
            mean_val = df[col].mean()
            if not np.isnan(mean_val):
                df[col] = df[col].fillna(mean_val)
            else:
                df[col] = df[col].fillna(0)

    df = df.astype(float)

    return df


def compute_features_from_signal(
    signal,
    sr,
    frame_length=FRAME_LENGTH,
    frame_period=FRAME_PERIOD,
    fft_length=N_FFT,
    n_channel=N_CHANNEL,
    sample_rate=SR,
    features_requested=None,
    device="cpu",
):
    """Compute SPTK features from a single audio signal.

    Args:
        signal: Audio signal as numpy array
        sr: Sampling rate of the signal
        frame_length: Frame length for STFT
        frame_period: Frame period for STFT
        fft_length: FFT length
        n_channel: Number of channels for filterbank
        sample_rate: Target sampling rate
        features_requested: List of features to extract
        device: Device to use for computation

    Returns:
        Dictionary of extracted features
    """
    if features_requested is None:
        features_requested = ["stft", "fbank"]

    # Initialize extractor
    extractor = SptkFeatureExtractor(
        frame_length=frame_length,
        frame_period=frame_period,
        fft_length=fft_length,
        n_channel=n_channel,
        sample_rate=sample_rate,
        device=device,
    )

    # Convert signal to tensor
    signal_tensor = torch.tensor(signal, device=device).float()

    # Extract features
    emb = extractor.extract_features_from_signal(signal_tensor, features_requested)

    return emb
