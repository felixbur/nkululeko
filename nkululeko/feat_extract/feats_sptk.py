"""Predict SPTK features.

https://github.com/sp-nitech/diffsptk

pip install diffsptk

"""

import os

import audiofile
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset
import diffsptk

# Constants
FRAME_LENGTH = 400  # Frame length.
FRAME_PERIOD = 80  # Frame period.
N_FFT = 512  # FFT length.
M_DIM = 24  # Mel-cepstrum dimensions.
SR = 16000  # Sampling rate.
N_CHANNEL = 128


class SptkSet(Featureset):
    """Class to predict SPTK features."""

    def __init__(self, name, data_df, feats_type):
        """Constructor.

        Is_train is needed to distinguish from test/dev sets,
        because they use the codebook from the training.
        """
        super().__init__(name, data_df, feats_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False

        # Read configuration from INI file with defaults
        self.frame_length = int(
            self.util.config_val("FEATS", "sptk.frame_length", FRAME_LENGTH)
        )
        self.frame_period = int(
            self.util.config_val("FEATS", "sptk.frame_period", FRAME_PERIOD)
        )
        self.fft_length = int(self.util.config_val("FEATS", "sptk.fft_length", N_FFT))
        self.n_channel = int(self.util.config_val("FEATS", "sptk.n_channel", N_CHANNEL))
        self.sample_rate = int(self.util.config_val("FEATS", "sptk.sample_rate", SR))

        self.stft = diffsptk.STFT(
            frame_length=self.frame_length,
            frame_period=self.frame_period,
            fft_length=self.fft_length,
            out_format="power",
        ).to(self.device)

        self.fbank = diffsptk.FBANK(
            fft_length=self.fft_length,
            n_channel=self.n_channel,
            sample_rate=self.sample_rate,
        ).to(self.device)

        self.mcep = diffsptk.MelCepstralAnalysis(
            fft_length=self.fft_length,
            cep_order=M_DIM,
            alpha=diffsptk.get_alpha(self.sample_rate),
            n_iter=10,
        ).to(self.device)

        # self.chroma = diffsptk.ChromaFilterBankAnalysis(
        #     self.fft_length, self.n_channel, self.sample_rate, device=self.device
        # )

        # Initialize pitch extractor with error handling for missing dependencies
        try:
            # Try default algorithm first (may require penn/crepe)
            self.pitch = diffsptk.Pitch(
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                f_min=80,
                f_max=180,
                voicing_threshold=0.4,
                out_format="f0",
            )
        except (ImportError, ModuleNotFoundError, AttributeError, Exception) as e:
            # Fall back to simpler pitch extraction using world vocoder
            self.util.debug(f"Could not initialize default pitch algorithm: {e}")
            self.util.debug("Using WORLD-based pitch extraction as fallback")
            # Use PyWorld-based pitch extraction instead
            import pyworld as pw

            self.pitch_fallback = True
            self.pw = pw
        else:
            self.pitch_fallback = False

        # Initialize pitch-dependent features only if pitch extraction works
        try:
            # aperiodicity extractor
            self.ap = diffsptk.Aperiodicity(
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                fft_length=self.fft_length,
                algorithm="d4c",
                out_format="a",
            )
            # spectral envelope extractor
            self.pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                fft_length=self.fft_length,
                algorithm="cheap-trick",
                out_format="power",
            )
            self.pitch_features_available = True
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            self.util.debug(f"Could not initialize pitch-dependent features: {e}")
            self.util.debug("Will extract only basic features (STFT, FBANK, MCEP)")
            self.pitch_features_available = False
            self.pitch_fallback = True  # Force using basic features only

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting SPTK, this might take a while...")
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            length = len(self.data_df.index)

            # Read audio file and extract features
            for idx, row_index in enumerate(
                tqdm(self.data_df.index.to_list(), total=length)
            ):
                # Handle both tuple (file, start, end) and string file formats
                if isinstance(row_index, tuple) and len(row_index) == 3:
                    file, start, end = row_index
                    signal, sampling_rate = audiofile.read(
                        file,
                        offset=start.total_seconds(),
                        duration=(end - start).total_seconds(),
                        always_2d=True,
                    )
                else:
                    # Single file path - read entire file
                    file = row_index if isinstance(row_index, str) else str(row_index)
                    signal, sampling_rate = audiofile.read(file, always_2d=True)

                # Extract features
                signal_tensor = torch.tensor(signal[0], device=self.device).float()

                # Pad signal if too short for frame_period
                if signal_tensor.shape[0] < self.frame_period:
                    pad_width = self.frame_period - signal_tensor.shape[0]
                    signal_tensor = torch.nn.functional.pad(
                        signal_tensor,
                        (0, pad_width),
                        mode="constant",
                        value=0,
                    )
                # Extract STFT first (needed for other features)
                stft_features = self.stft(signal_tensor)

                # Build feature dictionary with requested features only
                features_requested = self.util.config_val(
                    "FEATS", "features", "['stft', 'fbank']"
                )
                if isinstance(features_requested, str):
                    features_requested = eval(features_requested)

                emb = {}

                if "stft" in features_requested:
                    stft_np = stft_features.cpu().numpy()
                    # Flatten STFT to statistics over time frames
                    emb["stft_mean"] = np.mean(stft_np)
                    emb["stft_std"] = np.std(stft_np)
                    emb["stft_min"] = np.min(stft_np)
                    emb["stft_max"] = np.max(stft_np)

                if "fbank" in features_requested:
                    fbank_features = self.fbank(stft_features)
                    fbank_np = fbank_features.cpu().numpy()
                    # Flatten FBANK to per-channel statistics
                    for i in range(fbank_np.shape[-1]):
                        channel_data = fbank_np[..., i]
                        emb[f"fbank_{i}_mean"] = np.mean(channel_data)
                        emb[f"fbank_{i}_std"] = np.std(channel_data)

                if "mcep" in features_requested:
                    # MCEP expects power spectrum as input
                    mcep_features = self.mcep(stft_features)
                    mcep_np = mcep_features.cpu().numpy()
                    # Flatten MCEP to per-coefficient statistics
                    for i in range(mcep_np.shape[-1]):
                        coef_data = mcep_np[..., i]
                        emb[f"mcep_{i}_mean"] = np.mean(coef_data)
                        emb[f"mcep_{i}_std"] = np.std(coef_data)

                # Only extract pitch-dependent features if available
                if self.pitch_features_available and not (
                    hasattr(self, "pitch_fallback") and self.pitch_fallback
                ):
                    try:
                        # Handle pitch extraction with fallback
                        if hasattr(self, "pitch_fallback") and self.pitch_fallback:
                            # Use PyWorld for pitch extraction
                            signal_np = signal_tensor.cpu().numpy().astype(np.float64)
                            f0, _ = self.pw.dio(
                                signal_np,
                                self.sample_rate,
                                frame_period=self.frame_period
                                / self.sample_rate
                                * 1000,
                            )
                            f0 = self.pw.stonemask(signal_np, f0, _, self.sample_rate)
                            pitch_features = torch.tensor(f0, device=self.device)
                        else:
                            pitch_features = self.pitch(signal_tensor)

                        ap_features = self.ap(signal_tensor)
                        pitch_spec_features = self.pitch_spec(signal_tensor)

                        # Flatten pitch features to statistics
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
                    except Exception as e:
                        self.util.debug(
                            f"Warning: Could not extract pitch features for {file}: {e}"
                        )
                emb_series[row_index] = emb
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)

            # Fill NaN values with mean (similar to Praat)
            for col in self.df.columns:
                if self.df[col].isnull().values.any():
                    self.util.debug(
                        f"{col} includes {self.df[col].isnull().sum()} nan, inserting mean values"
                    )
                    self.df[col] = self.df[col].fillna(self.df[col].mean())

            # Convert to float like Praat does
            self.df = self.df.astype(float)

            # Print feature names if requested
            print_feats = self.util.config_val("FEATS", "print_feats", "False").strip().lower() == "true"
            if print_feats:
                self.util.debug(f"SPTK feature names: {self.df.columns.tolist()}")

            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted SPTK values")
            self.df = self.util.get_store(storage, store_format)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )
        return self.df

    def extract_sample(self, signal, sr):
        """Extract features from a single audio sample (like Praat's extract_sample)."""
        import audiofile
        import audformat

        tmp_audio_name = "sptk_audio_tmp.wav"
        audiofile.write(tmp_audio_name, signal, sr)
        df = pd.DataFrame(index=[tmp_audio_name])
        index = audformat.utils.to_segmented_index(df.index, allow_nat=False)

        # Extract features for this single sample
        signal_tensor = torch.tensor(signal, device=self.device).float()

        # Pad signal if too short
        if signal_tensor.shape[0] < self.frame_period:
            pad_width = self.frame_period - signal_tensor.shape[0]
            signal_tensor = torch.nn.functional.pad(
                signal_tensor, (0, pad_width), mode="constant", value=0
            )

        # Extract STFT
        stft_features = self.stft(signal_tensor)

        # Build features dictionary
        features_requested = self.util.config_val(
            "FEATS", "features", "['stft', 'fbank']"
        )
        if isinstance(features_requested, str):
            features_requested = eval(features_requested)

        emb = {}

        if "stft" in features_requested:
            stft_np = stft_features.cpu().numpy()
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

        # Create DataFrame and convert to numpy
        df = pd.DataFrame([emb])
        for col in df.columns:
            if df[col].isnull().values.any():
                mean_val = df[col].mean()
                if not np.isnan(mean_val):
                    df[col] = df[col].fillna(mean_val)
                else:
                    df[col] = df[col].fillna(0)

        df = df.astype(float)
        feats = df.to_numpy()

        # Clean up temporary file
        if os.path.exists(tmp_audio_name):
            os.remove(tmp_audio_name)

        return feats
