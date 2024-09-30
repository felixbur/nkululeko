# feats_ast.py
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import ASTModel, AutoProcessor

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class Ast(Featureset):
    """Class to extract AST (Audio Spectrogram Transformer) embeddings"""

    def __init__(self, name, data_df, feat_type):
        super().__init__(name, data_df, feat_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        self.feat_type = feat_type

    def init_model(self):
        self.util.debug("loading AST model...")
        model_path = self.util.config_val(
            "FEATS", "ast.model", "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = ASTModel.from_pretrained(model_path).to(self.device)
        print(f"initialized AST model on {self.device}")
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
                # make mono if stereo
                if signal.shape[0] == 2:
                    signal = torch.mean(signal, dim=0, keepdim=True)

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
        """Extract embeddings from raw audio signal."""
        try:
            inputs = self.processor(
                signal.numpy(), sampling_rate=sampling_rate, return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Get the hidden states
                outputs = self.model(**inputs)

            # Get the hidden states from the last layer
            last_hidden_state = outputs.last_hidden_state

            # print(f"last_hidden_state shape: {last_hidden_state.shape}")
            # Average pooling over the time dimension
            embeddings = torch.mean(last_hidden_state, dim=1)
            embeddings = embeddings.cpu().numpy()

            # print(f"hs shape: {embeddings.shape}")
            # hs shape: (1, 768)

        except Exception as e:
            self.util.error(
                f"Error extracting embeddings for file {file}: {str(e)}, fill with"
            )
            return np.zeros(
                self.model.config.hidden_size
            )  # Return zero vector on error
        return embeddings.ravel()

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
