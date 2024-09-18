# feats_wavlm.py
# HuBERT feature extractor for Nkululeko
# supported feat_types: 'wavlm-base', 'wavlm-base-plus', 'wavlm-large'

import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class Wavlm(Featureset):
    """Class to extract WavLM embedding)."""

    def __init__(self, name, data_df, feats_type):
        """Constructor.

        Is_train is needed to distinguish from test/dev sets,
        because they use the codebook from the training.
        """
        super().__init__(name, data_df, feats_type)
        # check if device is not set, use cuda if available
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        self.feat_type = feats_type

    def init_model(self):
        # load model
        self.util.debug("loading WavLM model...")

        model_path = self.util.config_val(
            "FEATS", "WavLM.model", f"microsoft/{self.feat_type}"
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = WavLMModel.from_pretrained(model_path).to(self.device)
        print(f"intialized WavLM model on {self.device}")
        self.model.eval()
        self.model_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug("extracting wavlm embeddings, this might take a while...")
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = torchaudio.load(
                    file,
                    frame_offset=int(start.total_seconds() * 16000),
                    num_frames=int((end - start).total_seconds() * 16000),
                )
                assert (
                    sampling_rate == 16000
                ), f"sampling rate should be 16000 but is {sampling_rate}"
                emb = self.get_embeddings(signal, sampling_rate, file)
                emb_series.iloc[idx] = emb
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)
            self.df.to_pickle(storage)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted {self.feat_type} embeddings")
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                # nanrows = self.df.columns[self.df.isna().any()].tolist()
                # print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, signal, sampling_rate, file):
        r"""Extract embeddings from raw audio signal."""
        if sampling_rate != 16000:
            self.util.error(f"sampling rate should be 16000 but is {sampling_rate}")
        try:
            with torch.no_grad():
                # run through processor to normalize signal
                # always returns a batch, so we just get the first entry
                # then we put it on the device
                y = self.processor(signal, sampling_rate=sampling_rate)
                y = y["input_values"][0]
                y = torch.from_numpy(y.reshape(1, -1)).to(self.device)
                # run through model
                # first entry contains hidden state
                y = self.model(y)[0]

                # pool result and convert to numpy
                y = torch.mean(y, dim=1)
                y = y.detach().cpu().numpy()

                # print(f"hs shape: {y.shape}")

        except RuntimeError as re:
            print(str(re))
            self.util.error(f"Couldn't extract file: {file}")

        return y.ravel()

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
