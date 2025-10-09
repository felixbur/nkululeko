"""Predict SPTK features.

https://github.com/sp-nitech/diffsptk

pip install diffsptk

"""

import os

import numpy as np
import pandas as pd
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset
from nkululeko.feat_extract import feats_sptk_core

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

        # Get requested features from config
        self.features_requested = self.util.config_val(
            "FEATS", "sptk.features", "['stft', 'fbank']"
        )
        if isinstance(self.features_requested, str):
            self.features_requested = eval(self.features_requested)

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting SPTK, this might take a while...")

            # Use core module for feature extraction
            self.df = feats_sptk_core.compute_features(
                file_index=self.data_df.index,
                frame_length=self.frame_length,
                frame_period=self.frame_period,
                fft_length=self.fft_length,
                n_channel=self.n_channel,
                sample_rate=self.sample_rate,
                features_requested=self.features_requested,
                device=self.device,
            )

            # Print feature names if requested
            print_feats = (
                self.util.config_val("FEATS", "print_feats", "False").strip().lower()
                == "true"
            )
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
        # Use core module for feature extraction
        emb = feats_sptk_core.compute_features_from_signal(
            signal=signal,
            sr=sr,
            frame_length=self.frame_length,
            frame_period=self.frame_period,
            fft_length=self.fft_length,
            n_channel=self.n_channel,
            sample_rate=self.sample_rate,
            features_requested=self.features_requested,
            device=self.device,
        )

        # Convert to DataFrame and then to numpy
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

        return feats
