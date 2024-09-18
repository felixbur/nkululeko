# feats_whisper.py
import os

import audeer
import audiofile
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WhisperModel

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class Whisper(Featureset):
    """Class to extract whisper embeddings."""

    def __init__(self, name, data_df, feat_type):
        super().__init__(name, data_df, feat_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        if feat_type == "whisper":
            self.feat_type = "whisper-base"
        else:
            self.feat_type = feat_type

    def init_model(self):
        # load model
        self.util.debug("loading whisper model...")
        model_name = f"openai/{self.feat_type}"
        self.model = WhisperModel.from_pretrained(model_name).to(self.device)
        print(f"intialized Whisper model on {self.device}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
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
            self.util.debug("extracting whisper embeddings, this might take a while...")
            emb_series = []
            for (file, start, end), _ in audeer.progress_bar(
                self.data_df.iterrows(),
                total=len(self.data_df),
                desc=f"Running whisper on {len(self.data_df)} audiofiles",
            ):
                if end == pd.NaT:
                    signal, sr = audiofile.read(file, offset=start)
                else:
                    signal, sr = audiofile.read(
                        file, duration=end - start, offset=start
                    )
                emb = self.get_embeddings(signal, sr, file)
                emb_series.append(emb)
            # print(f"emb_series shape: {emb_series.shape}")
            self.df = pd.DataFrame(emb_series, index=self.data_df.index)
            # print(f"df shape: {self.df.shape}")
            self.df.to_pickle(storage)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted wav2vec2 embeddings")
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                # print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, signal, sampling_rate, file):
        r"""Extract embeddings from raw audio signal."""
        try:
            with torch.no_grad():
                embed_size = self.model.config.hidden_size
                embed_columns = [f"whisper_{i}" for i in range(embed_size)]
                inputs = self.feature_extractor(signal, sampling_rate=16000)[
                    "input_features"
                ][0]
                inputs = torch.from_numpy(inputs).to(self.device).unsqueeze(0)
                decoder_input_ids = (
                    torch.tensor([[1, 1]]).to(self.device)
                    * self.model.config.decoder_start_token_id
                )
                full_outputs = self.model(
                    inputs,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                )
                outputs = full_outputs.encoder_last_hidden_state[0]
                average_embeds = outputs.squeeze().mean(axis=0).cpu().detach().numpy()
        except RuntimeError as re:
            print(str(re))
            self.util.error(f"couldn't extract file: {file}")
        # print(f"y flattened shape: {y.ravel().shape}")
        return average_embeds

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
