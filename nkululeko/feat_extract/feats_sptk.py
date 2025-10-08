"""Predict SPTK features.

https://github.com/sp-nitech/diffsptk

pip install diffsptk

"""

import os

import audiofile
import pandas as pd
from scipy import signal
from tqdm import tqdm
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset
import diffsptk


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
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list(), total=length)
            ):
                signal, sampling_rate = audiofile.read(
                    file,
                    offset=start.total_seconds(),
                    duration=(end - start).total_seconds(),
                    always_2d=True,
                )
                emb = self.get_extract(signal, sampling_rate, file)
                emb_series[idx] = emb
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)
            self.df.columns = ["pesq", "sdr", "stoi"]
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

    def get_extract(self, signal, sampling_rate, file):
        # mc = self.get_melcepstrum(signal, sampling_rate)
        f0 = self.get_pitch(signal, sampling_rate)
        # chroma = self.get_chroma(signal, sampling_rate)
        # return mc

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_extract(signal, sr, "no file")
        return feats

    def get_melcepstrum(self, signal, sr):
        """Get melcepstrum features from signal."""
        fl = 400  # Frame length.
        fp = 80  # Frame period.
        n_fft = 512  # FFT length.
        M = 24  # Mel-cepstrum dimensions.# Compute STFT amplitude of x.
        stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)
        tensor = torch.from_numpy(signal)
        X = stft(tensor).to(self.device)

        # Estimate mel-cepstrum of x.
        alpha = diffsptk.get_alpha(sr)
        mcep = diffsptk.MelCepstralAnalysis(
            fft_length=n_fft,
            cep_order=M,
            alpha=alpha,
            n_iter=10,
        )
        mc = mcep(X)
        return mc.detach().numpy().flatten()

    def get_pitch(self, signal, sr):
        """Get pitch from signal."""

        tensor = torch.from_numpy(signal)
        fp = 80  # Frame period.
        n_fft = 1024  # FFT length.

        # Extract F0 of x, or prepare well-estimated F0.
        pitch = diffsptk.Pitch(
            frame_period=fp,
            sample_rate=sr,
            f_min=80,
            f_max=180,
            voicing_threshold=0.4,
            out_format="f0",
        )
        f0 = pitch(x).to(self.device)
        # Extract aperiodicity of x by D4C.
        ap = diffsptk.Aperiodicity(
            frame_period=fp,
            sample_rate=sr,
            fft_length=n_fft,
            algorithm="d4c",
            out_format="a",
        )
        A = ap(tensor, f0)

        # Extract spectral envelope of x by CheapTrick.
        pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(
            frame_period=fp,
            sample_rate=sr,
            fft_length=n_fft,
            algorithm="cheap-trick",
            out_format="power",
        )
        S = pitch_spec(tensor, f0)
        return f0, A, S

    def get_chroma(self, signal, sr):
        tensor = torch.from_numpy(signal)
        stft = diffsptk.STFT(frame_length=512, frame_period=512, fft_length=512)

        chroma = diffsptk.ChromaFilterBankAnalysis(12, 512, sr, device=self.device)

        y = chroma(stft(tensor)).to(self.device)
        return y.detach().numpy().flatten()
