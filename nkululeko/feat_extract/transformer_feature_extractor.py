# transformer_feature_extractor.py

import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class TransformerFeatureExtractor(Featureset):
    def __init__(self, name, data_df, feat_type):
        super().__init__(name, data_df, feat_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        self.feat_type = feat_type

    def init_model(self):
        raise NotImplementedError("Subclasses must implement init_model method")

    def get_embeddings(self, signal, sampling_rate, file):
        try:
            with torch.no_grad():
                # Preprocess the input
                inputs = self.preprocess_input(signal, sampling_rate)

                # Get model outputs
                outputs = self.model(**inputs)

                # Extract the relevant hidden states
                hidden_states = self.extract_hidden_states(outputs)

                # Pool the hidden states
                embeddings = self.pool_hidden_states(hidden_states)

                # Convert to numpy and flatten
                embeddings = embeddings.cpu().numpy().ravel()

            return embeddings

        except Exception as e:
            self.util.error(f"Error extracting embeddings for file {file}: {str(e)}")
            return np.zeros(self.get_embedding_dim())  # Return zero vector on error

    def preprocess_input(self, signal, sampling_rate):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement preprocess_input method")

    def extract_hidden_states(self, outputs):
        # This method should be implemented by subclasses
        raise NotImplementedError(
            "Subclasses must implement extract_hidden_states method"
        )

    def pool_hidden_states(self, hidden_states):
        # Default implementation: mean pooling over time dimension
        return torch.mean(hidden_states, dim=1)

    def get_embedding_dim(self):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_embedding_dim method")

    def extract(self):
        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                f"extracting {self.feat_type} embeddings, this might take a while..."
            )
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = torchaudio.load(
                    file,
                    frame_offset=int(start.total_seconds() * 16000),
                    num_frames=int((end - start).total_seconds() * 16000),
                )
                assert sampling_rate == 16000, f"got {sampling_rate} instead of 16000"
                emb = self.get_embeddings(signal, sampling_rate, file)
                emb_series[idx] = emb
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
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats

        # for each feature extractor
        # feats_ast.py

        # class Ast(TransformerFeatureExtractor):
        #     def preprocess_input(self, signal, sampling_rate):
        #         inputs = self.processor(signal.numpy(), sampling_rate=sampling_rate, return_tensors="pt")
        #         return {k: v.to(self.device) for k, v in inputs.items()}

        #     def extract_hidden_states(self, outputs):
        #         return outputs.last_hidden_state

        #     def get_embedding_dim(self):
        #         return self.model.config.hidden_size

        # # feats_wav2vec2.py

        # class Wav2vec2(TransformerFeatureExtractor):
        #     def preprocess_input(self, signal, sampling_rate):
        #         inputs = self.processor(signal, sampling_rate=sampling_rate, return_tensors="pt")
        #         return {k: v.to(self.device) for k, v in inputs.items()}

        #     def extract_hidden_states(self, outputs):
        #         return outputs.last_hidden_state

        #     def get_embedding_dim(self):
        #         return self.model.config.hidden_size

        # # feats_wavlm.py

        # class Wavlm(TransformerFeatureExtractor):
        #     def preprocess_input(self, signal, sampling_rate):
        #         inputs = self.processor(signal, sampling_rate=sampling_rate, return_tensors="pt")
        #         return {k: v.to(self.device) for k, v in inputs.items()}

        #     def extract_hidden_states(self, outputs):
        return outputs.last_hidden_state

    # def get_embedding_dim(self):
    #     return self.model.config.hidden_size
